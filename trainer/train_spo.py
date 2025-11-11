import os
import sys
from typing import cast 

# Package path manipulation to ensure proper imports from parent directory
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, 
    setup_seed, SkipBatchSampler, init_model
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


class AutoAdaptiveValueTracker:
    """
    SPO (Self-Play Optimization) Adaptive Value Tracker.
    
    This class implements an adaptive baseline mechanism that maintains running
    statistics (alpha/beta parameters) to estimate baselines without a separate
    critic model. The baseline adapts based on the mean log probability of
    responses, using a KL-based "rho" parameter to control the update rate.
    
    Key Features:
        - No separate critic model needed (memory efficient)
        - Adaptive baseline based on policy behavior
        - KL-based rho parameter for controlled updates
        - Clipped updates for stability
    
    Mathematical Formulation:
        - Baseline = alpha / (alpha + beta)  [expectation in [0,1]]
        - Rho = 2^(-KL / D_half)  [adaptive update rate]
        - Alpha/Beta updated as: alpha = rho*alpha + reward, beta = rho*beta + (1-reward)
    
    Args:
        rho_mode: Method to compute rho ('constant' or 'kl' for KL-based)
        rho_const: Constant rho value when rho_mode='constant'
        D_half: KL half-life for rho computation (smaller = faster decay)
        clip_lower: Lower bound for rho clipping
        clip_upper: Upper bound for rho clipping
    """
    def __init__(
        self, 
        rho_mode: str = 'kl', 
        rho_const: float = 0.9, 
        D_half: float = 0.06, 
        clip_lower: float = 0.5, 
        clip_upper: float = 0.96
    ):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        
        # Initialize alpha/beta parameters for beta distribution
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init
        self.beta = 0.5 * N_init
        
        # Track old mean log probability for KL computation
        self.old_mean_logprob = None

    def get_baselines(self, batch_size: int) -> torch.Tensor:
        """
        Compute baseline values for the current batch.
        
        Baselines are sampled from the current beta distribution estimate
        and returned as a tensor of shape [batch_size].
        
        Args:
            batch_size: Number of samples in the batch
            
        Returns:
            Baseline tensor [batch_size] in range [0, 1]
        """
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob: float) -> float:
        """
        Compute adaptive rho parameter based on KL divergence.
        
        Rho controls how quickly the baseline adapts to new data.
        - When policy changes little (small KL), rho stays high (fast updates)
        - When policy changes significantly (large KL), rho drops (slow updates)
        
        Args:
            cur_mean_logprob: Current mean log probability
            
        Returns:
            Rho value in range [clip_lower, clip_upper]
        """
        if self.rho_mode == 'constant':
            return self.rho_const
        
        # First iteration: use constant rho
        if self.old_mean_logprob is None:
            return self.rho_const
        
        # Compute KL divergence from old policy
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        
        # Compute rho as exponential decay based on KL
        rho = 2 ** (-kl / self.D_half)
        
        # Clip to stable range
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(
        self, 
        rewards: torch.Tensor, 
        cur_logprobs: torch.Tensor, 
        response_masks: torch.Tensor
    ) -> float:
        """
        Update the value tracker with new rewards and log probabilities.
        
        This method:
        1. Computes rho (adaptive update rate)
        2. Normalizes rewards to [0,1] range
        3. Updates alpha/beta parameters using rho-weighted averages
        4. Returns the computed rho value
        
        Args:
            rewards: Raw reward values [B]
            cur_logprobs: Current log probabilities [B, seq_len]
            response_masks: Mask for valid response tokens [B, seq_len]
            
        Returns:
            Computed rho value for logging
        """
        # Compute mean log probability for KL-based rho
        if cur_logprobs is not None and response_masks is not None:
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        # Normalize rewards to [0, 1] range for beta distribution update
        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        
        # Update alpha/beta with rho-weighted running averages
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        
        return rho


def calculate_rewards(
    prompts: list[str], 
    responses: list[str], 
    reward_model, 
    reward_tokenizer
) -> torch.Tensor:
    """
    Compute comprehensive reward scores for generated responses.
    
    This function integrates multiple reward signals:
    1. **Format Rewards**: For reasoning models, validates `<think>...</think><answer>...</answer>` structure
    2. **Tag Counting Rewards**: Ensures proper usage of reasoning tags
    3. **Reward Model Scores**: Uses external reward model to evaluate response quality
    
    Args:
        prompts: List of input prompts (batch_size)
        responses: List of generated responses (batch_size)
        reward_model: External reward model for scoring response quality
        reward_tokenizer: Tokenizer for the reward model
        
    Returns:
        Tensor of reward scores of shape (batch_size,)
    
    Reward Calculation:
        - Format reward: 0.5 for correct structure, 0.0 otherwise
        - Tag counting: 0.25 per correctly used tag (max 1.0)
        - Reward model: Scores clipped to [-scale, scale] range
        - For reasoning models: Combines full response (40%) and answer content (60%)
    """
    def reasoning_model_reward(rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute format-based rewards for reasoning model responses.
        
        Validates strict `<think>...</think><answer>...</answer>` format and counts
        correct tag usage to prevent sparse rewards during training.
        """
        # Pattern 1: Exact format without extra newlines
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # Pattern 2: Format with extra newline between sections
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        # Check response format
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # Assign format rewards
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text: str) -> float:
            """Count correctly used reasoning tags (0.25 reward per tag)."""
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        # Count tag usage rewards
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # Initialize rewards tensor
    rewards = torch.zeros(len(responses), device=args.device)

    # Apply reasoning-specific rewards if in reasoning mode
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # Compute reward model scores
    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0  # Clipping range

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Parse prompt into role-based message format
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # Score the full response
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            score = max(min(score, scale), -scale)

            # For reasoning models: also score extracted answer content separately
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # Score answer content separately
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # Combine scores: 40% full response, 60% answer content
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        # Convert to tensor and add to rewards
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def spo_train_epoch(
    epoch: int,
    loader: DataLoader,
    iters: int,
    ref_model: MiniMindForCausalLM,
    reward_model,
    reward_tokenizer,
    value_tracker: AutoAdaptiveValueTracker,
    start_step: int = 0,
    wandb=None
) -> None:
    """
    Train model for one epoch using Self-Play Optimization (SPO).
    
    SPO is a reinforcement learning algorithm that uses an adaptive value tracker
    instead of a separate critic model. It maintains running statistics (alpha/beta)
    to estimate baselines, which are updated based on policy behavior via a KL-based
    rho parameter.
    
    Key Differences from PPO/GRPO:
        - No separate critic model (memory efficient)
        - Adaptive baseline based on policy log probabilities
        - Single generation per prompt (not multiple)
        - Direct advantage computation without normalization
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing RL training data
        iters: Total iterations in this epoch
        ref_model: Frozen reference model for KL penalty
        reward_model: External reward model for scoring responses
        reward_tokenizer: Tokenizer for reward model
        value_tracker: Adaptive baseline estimator
        start_step: Starting step if resuming from checkpoint
        wandb: Experiment tracking logger (or None)
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B (batch_size)
        
        # Tokenize prompts with left-padding for generation
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False,
            padding_side="left", 
            add_special_tokens=False
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # Truncate to maximum sequence length if specified
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ========== Generation Phase ==========
        with torch.no_grad():
            # DDP models need .module access for generate method
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            
            # Generate single response per prompt (SPO uses self-play, not group comparison)
            outputs = model_for_gen.generate(
                **prompt_inputs, 
                max_new_tokens=args.max_gen_len, 
                do_sample=True, 
                temperature=0.8,
                num_return_sequences=1, 
                pad_token_id=tokenizer.pad_token_id
            )  # [B, P+R]

        # Extract completion tokens (remove prompt)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B, R]

        # ========== Log Probability Calculation ==========
        def get_per_token_logps(mdl, input_ids: torch.Tensor, n_keep: int) -> torch.Tensor:
            """
            Compute per-token log probabilities for the last n_keep tokens.
            
            Uses logits_to_keep for efficiency when only the last tokens' logits
            are needed for loss computation.
            """
            # Clone to avoid modifying original tensor
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            
            # Forward pass with logits_to_keep for efficiency
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            
            # Gather log probabilities for target tokens
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(
                    torch.gather(
                        logits_row.log_softmax(dim=-1), 
                        1, 
                        ids_row.unsqueeze(1)
                    ).squeeze(1)
                )
            return torch.stack(per_token_logps)

        # Get log probabilities from policy and reference models
        outputs = cast(torch.Tensor, outputs)
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B, R]
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B, R]

        # ========== Reward Calculation ==========
        # Decode completions to text for reward calculation
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)  # list[str], length B
        
        # Calculate rewards using external reward model
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B]

        # ========== Baseline and Advantage Computation ==========
        # Get baselines from value tracker (in [0,1] range)
        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)  # [B]
        
        # Un-normalize baselines to match raw reward scale [-scale, scale]
        scale = 3.0
        unnormalized_baselines = baselines * (2 * scale) - scale  # [B]
        
        # Compute advantages (reward - baseline)
        advantages = rewards - unnormalized_baselines  # [B]

        # Use baseline-provided advantages, only clip for stability.
        # No batch normalization because baseline already provides stable cross-batch estimates.
        advantages = advantages.clamp(-5.0, 5.0)

        # ========== Completion Mask Creation ==========
        # Create mask to ignore tokens after EOS
        is_eos = completion_ids == tokenizer.eos_token_id  # [B, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)  # [B]
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # Binary mask: 1 for tokens before/inclusive of EOS, 0 after
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= 
            eos_idx.unsqueeze(1)
        ).int()  # [B, R]

        # ========== Policy Loss Computation ==========
        # KL divergence between policy and reference
        kl_div = ref_per_token_logps - per_token_logps  # [B, R]
        
        # Per-token KL penalty
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B, R]
        
        # SPO loss: advantage-weighted policy loss with KL penalty
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl  # [B, R]
        
        # Average loss over non-padded tokens, then over batch
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / 
            completion_mask.sum(dim=1)
        ).mean() / args.accumulation_steps  # scalar
        loss.backward()

        # ========== Value Tracker Update ==========
        # Update alpha/beta parameters based on rewards and log probabilities
        response_masks = completion_mask.float()  # [B, R]
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)

        # ========== Optimization Step ==========
        if (step + 1) % args.accumulation_steps == 0:
            # Gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step and scheduler update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # ========== Logging & Monitoring ==========
        if step % args.log_interval == 0 or step == iters:
            # Extract metrics
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            # Print log message
            Logger(
                f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}, '
                f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}'
            )

            # Log to wandb
            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # ========== Checkpointing ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract state dict (handle DDP wrapper)
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            
            # Save in half precision
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            
            # Save full checkpoint with optimizer and scheduler states
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', 
                scheduler=scheduler
            )
            model.train()

        # ========== Memory Cleanup ==========
        # Explicitly delete tensors and collect garbage due to memory-intensive RL nature
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    """
    Main training script for MiniMind Self-Play Optimization (SPO).
    
    SPO is a reinforcement learning algorithm that uses an adaptive value tracker
    instead of a separate critic model. It maintains running statistics (alpha/beta
    parameters) to estimate baselines, which are updated based on policy behavior
    via a KL-based rho parameter.
    
    Key SPO Characteristics:
        - Memory efficient (no critic model)
        - Adaptive baseline based on policy log probabilities
        - Single generation per prompt (self-play style)
        - KL-controlled update rate for stability
    
    Training Process:
        1. Generate single response per prompt
        2. Calculate rewards using reward model
        3. Get baselines from adaptive value tracker
        4. Compute advantages (reward - baseline)
        5. Update policy with KL penalty
        6. Update value tracker statistics
    """
    parser = argparse.ArgumentParser(description="MiniMind SPO (Self-Play Optimization)")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='spo', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=10, help="Model saving interval (steps)")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Generation parameters
    parser.add_argument('--max_seq_len', default=66, type=int, help="Maximum prompt length")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="Maximum generation length")
    
    # RL parameters
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='Model type (0=regular, 1=reasoning)')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward model path")
    
    # Data and resumption
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF training data path")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SPO", help="wandb project name")
    
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    # Set up distributed training if available (multi-GPU)
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # Set random seed for reproducibility, different per process
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Configure directories, model parameters, check checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize model configuration (account for generation length)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len, 
        use_moe=bool(args.use_moe)
    )
    
    # Check for existing checkpoint if resuming
    ckp_data = lm_checkpoint(
        lm_config, weight=args.save_weight, save_dir='../checkpoints'
    ) if args.from_resume == 1 else None
    
    # ========== 3. Set up mixed precision training ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # No-op context for CPU, autocast for CUDA
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. Configure wandb logging ==========
    # Note: This imports swanlab as wandb, which is a Chinese experiment tracking platform
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize models and value tracker ==========
    # Base weight to initialize from (reasoning or full_sft)
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Policy model (being trained)
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Reference model (frozen, for KL penalty)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward model (external model for scoring responses)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # Adaptive value tracker (SPO's unique component)
    value_tracker = AutoAdaptiveValueTracker(
        rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96
    )
    
    # Training dataset and optimizer
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Count total iterations for learning rate scheduler
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    
    # Cosine annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. Restore training state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap model with DistributedDataParallel ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization across GPUs
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore 
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Start main training loop ==========
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler to ensure proper data shuffling
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Handle checkpoint resumption for first epoch
        if epoch == start_epoch and start_step > 0:
            # Skip already processed batches
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps, '
                f'starting from step {start_step + 1}'
            )
            spo_train_epoch(
                epoch, loader, len(loader) + start_step + 1, ref_model, 
                reward_model, reward_tokenizer, value_tracker, start_step, wandb
            )
        else:
            # Standard dataloader for normal training
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                pin_memory=True,
                drop_last=False, 
                shuffle=(train_sampler is None),
                num_workers=args.num_workers, 
                sampler=train_sampler
            )
            spo_train_epoch(
                epoch, loader, len(loader), ref_model, 
                reward_model, reward_tokenizer, value_tracker, 0, wandb
            )