import os
import sys
from typing import cast 

# Package path manipulation to ensure proper imports from parent directory
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, 
    setup_seed, SkipBatchSampler, init_model, calculate_mfu
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


class CriticModel(MiniMindForCausalLM):
    """
    Custom Critic model for PPO value function estimation.
    
    This model inherits from MiniMindForCausalLM but replaces the language modeling
    head with a value head that outputs a scalar value estimate for each token.
    The critic provides state value estimates used for advantage computation in PPO.
    
    Architecture:
        - Base: MiniMind transformer for encoding sequences
        - Head: Single linear layer (hidden_size -> 1) for value prediction
    """
    def __init__(self, params: MiniMindConfig):
        super().__init__(params)
        # Replace lm_head with a value head that outputs scalar value estimates
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs): # NOTE: Check this! #type: ignore 
        """
        Forward pass to compute state values.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            values: Scalar value estimates [batch_size, seq_len]
        """
        # Get hidden states from base model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        
        # Compute values using value head
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(
    prompts: list[str], 
    responses: list[str], 
    reward_model, 
    reward_tokenizer
) -> torch.Tensor:
    """
    Compute comprehensive reward scores for generated responses.
    
    This function integrates multiple reward signals for RL training:
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
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
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
        for prompt, response in zip(prompts, responses):
            # Parse prompt into role-based message format
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # Score the full response
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # Clip score to prevent extreme values
            scale = 3.0
            score = max(min(score, scale), -scale)

            # For reasoning models: also score extracted answer content
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


def ppo_train_epoch(
    epoch: int,
    loader: DataLoader,
    iters: int,
    old_actor_model: MiniMindForCausalLM,
    ref_model: MiniMindForCausalLM,
    actor_scheduler,
    critic_scheduler,
    reward_model,
    reward_tokenizer,
    start_step: int = 0,
    wandb=None
) -> None:
    """
    Train models for one epoch using Proximal Policy Optimization (PPO).
    
    PPO is a state-of-the-art reinforcement learning algorithm for language model alignment.
    It uses an actor-critic architecture with:
    - Actor model: Generates responses (policy)
    - Critic model: Estimates state values
    - Old actor: For importance sampling ratio computation
    - Reference model: For KL penalty to prevent policy collapse
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing RL training data
        iters: Total iterations in this epoch
        old_actor_model: Snapshot of actor for policy ratio computation
        ref_model: Frozen reference model for KL penalty
        actor_scheduler: Learning rate scheduler for actor
        critic_scheduler: Learning rate scheduler for critic
        reward_model: External reward model for scoring responses
        reward_tokenizer: Tokenizer for reward model
        start_step: Starting step if resuming
        wandb: Experiment tracking logger
    
    PPO Features:
        - Clipped policy objective for stable updates
        - Separate value function learning (critic)
        - KL divergence penalty against reference policy
        - Per-token log probability computation
        - Periodic old actor updates
    """
    # Set models to training mode
    actor_model.train()
    critic_model.train()
    
    # Initialize tracking variables for metrics
    import time
    start_time = time.time()
    total_tokens_seen = 0
    grad_norm_actor = 0.0
    grad_norm_critic = 0.0
    
    # Calculate model FLOPs for MFU calculation
    model_flops_per_token = 6 * sum(p.numel() for p in actor_model.parameters() if p.requires_grad)

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], length B
        
        # Tokenize prompts
        enc = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=args.max_seq_len
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        prompt_lengths = enc.attention_mask.sum(dim=1)  # [B]

        # ========== Generation Phase ==========
        with torch.no_grad():
            # DDP models need .module access for generate method
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            
            # Generate responses
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, 
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len, 
                do_sample=True, 
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )  # [B, P+R]

        # Extract response text and compute rewards
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_lengths[i]:], skip_special_tokens=True) 
            for i in range(len(prompts))
        ]
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # ========== Value Estimation ==========
        # Create mask for non-padded tokens
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        
        # Get value estimates from critic
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        
        # Extract final value for each sequence (at EOS position)
        last_indices = full_mask.sum(dim=1) - 1  # [B]
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        
        # Compute advantages (reward - value)
        advantages = rewards - values.detach()  # [B]

        # ========== Log Probability Computation ==========
        # Get log probabilities from actor model
        logits = actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        
        # Compute per-token log probabilities
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        # Create mask for response tokens only (exclude prompt and padding)
        gen_out = cast(torch.Tensor, gen_out) # NOTE: Check this!
        seq_len = gen_out.size(1) - 1 
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        
        # Sum log probabilities over response tokens
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== Old Actor & Reference Log Probabilities ==========
        with torch.no_grad():
            # Old actor log probabilities (for importance ratio)
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # Reference model log probabilities (for KL penalty)
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits  # [B, P+R, V]
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== Policy Loss (Clipped Objective) ==========
        # KL divergences for monitoring
        kl = (actor_logp - old_logp).mean()  # scalar
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        
        # Policy ratio for importance sampling
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # Clipped surrogate objective
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        
        # Value function loss (MSE)
        value_loss = F.mse_loss(values, rewards)  # scalar
        
        # Total loss with KL penalty and value coefficient
        loss = policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref  # scalar
        loss.backward()
        
        # Track tokens seen (response tokens)
        total_tokens_seen += (gen_out.shape[1] - enc.input_ids.shape[1]) * gen_out.shape[0]

        # ========== Optimization Step ==========
        if (step + 1) % args.accumulation_steps == 0:
            # Gradient clipping for both actor and critic
            grad_norm_actor = clip_grad_norm_(actor_model.parameters(), args.grad_clip).item()
            grad_norm_critic = clip_grad_norm_(critic_model.parameters(), args.grad_clip).item()
            
            # Optimizer steps
            actor_optimizer.step()
            critic_optimizer.step()
            
            # Scheduler steps
            actor_scheduler.step()
            critic_scheduler.step()
            
            # Zero gradients
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            torch.cuda.empty_cache()

        # ========== Logging & Monitoring ==========
        if is_main_process():
            # Compute average response length
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(
                has_eos, 
                eos_indices + 1, 
                torch.tensor(response_ids.shape[1], device=is_eos.device)
            )
            avg_len = lengths.float().mean()

            # Extract metrics
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']
            
            # Calculate throughput metrics
            spend_time = time.time() - start_time
            tokens_per_sec = total_tokens_seen / spend_time if spend_time > 0 else 0
            samples_per_sec = (step * args.batch_size) / spend_time if spend_time > 0 else 0
            steps_per_sec = step / spend_time if spend_time > 0 else 0
            
            # Calculate MFU (Model FLOPs Utilization)
            mfu = calculate_mfu(model_flops_per_token, tokens_per_sec, args.device)
            
            # Global step across all epochs
            global_step = epoch * iters + step
            
            # Estimated time remaining
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            # Log to wandb with comprehensive metrics
            if wandb is not None:
                log_dict = {
                    # PPO-specific metrics
                    "train/actor_loss": actor_loss_val,
                    "train/critic_loss": critic_loss_val,
                    "train/reward": reward_val,
                    "train/kl": kl_val,
                    "train/kl_ref": kl_ref_val,
                    "train/avg_response_len": avg_len_val,
                    
                    # Learning rates
                    "train/actor_lr": actor_lr,
                    "train/critic_lr": critic_lr,
                    
                    # Gradient norms
                    "train/actor_grad_norm": grad_norm_actor,
                    "train/critic_grad_norm": grad_norm_critic,
                    
                    # Progress tracking
                    "train/epoch": epoch + (step / iters),
                    "train/global_step": global_step,
                    
                    # Throughput metrics
                    "train/tokens_per_second": tokens_per_sec,
                    "train/samples_per_second": samples_per_sec,
                    "train/steps_per_second": steps_per_sec,
                    "train/num_tokens_generated": total_tokens_seen,
                    
                    # Efficiency metrics
                    "train/mfu_percent": mfu,
                    "train/eta_minutes": eta_min,
                    
                    # System metrics
                    "system/epoch_time": spend_time / 60,  # in minutes
                }
                wandb.log(log_dict, step=global_step)

            # Enhanced print log message
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) "
                f"actor_loss:{actor_loss_val:.6f} critic_loss:{critic_loss_val:.6f} "
                f"reward:{reward_val:.4f} kl:{kl_val:.4f} "
                f"grad_norm_actor:{grad_norm_actor:.4f} tokens/s:{tokens_per_sec:.0f} "
                f"MFU:{mfu:.2f}% epoch_Time:{eta_min}min"
            )

        # ========== Old Actor Update ==========
        # Periodically update old actor model for importance sampling
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ========== Checkpointing ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract actor state dict (handle DDP wrapper)
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            
            # Save actor weights in half precision
            torch.save({k: v.half() for k, v in actor_state.items()}, ckp)
            
            # Save full checkpoint with all components
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model, 
                optimizer=actor_optimizer, epoch=epoch, step=step, wandb=wandb, 
                save_dir='checkpoints', scheduler=actor_scheduler, 
                critic_model=critic_model, critic_optimizer=critic_optimizer, 
                critic_scheduler=critic_scheduler
            )
            actor_model.train()


if __name__ == "__main__":
    """
    Main training script for MiniMind Proximal Policy Optimization (PPO).
    
    PPO is a reinforcement learning algorithm for aligning language models with
    human preferences using an actor-critic architecture. It maintains:
    - Actor model: The policy being optimized (generates responses)
    - Old actor: Snapshot for importance sampling ratio computation
    - Critic model: Value function for advantage estimation
    - Reference model: Frozen model for KL penalty
    
    This implementation includes special support for reasoning models with structured
    output format validation and reward shaping.
    
    Key PPO Features:
        - Clipped policy objective for stable updates
        - Separate value function learning
        - KL divergence penalty against reference policy
        - Importance sampling with old policy
        - Periodic old actor updates
    """
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="out", help="Model save directory")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (number of prompts)")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor learning rate")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=1, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=10, help="Model saving interval (steps)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval (steps), 0 to disable")
    parser.add_argument("--eval_batches", type=int, default=50, help="Number of batches to use for evaluation")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Generation parameters
    parser.add_argument('--max_seq_len', default=66, type=int, help="Maximum prompt length")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="Maximum generation length")
    
    # RL parameters
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO clipping parameter epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL divergence penalty coefficient")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='Model type (0=regular, 1=reasoning)')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="Frequency to update old actor model")
    parser.add_argument("--reward_model_path", type=str, default="internlm/internlm2-1_8b-reward", help="Reward model path") # NOTE: Using the Hugginggface path
    
    # Data and resumption
    parser.add_argument("--data_path", type=str, default="dataset/rlhf_ppo_train.jsonl", help="RLAIF training data path")
    parser.add_argument("--data_config", type=str, default=None, help="Path to dataset mixture YAML config")
    parser.add_argument("--use_prepared", action="store_true", help="Use pre-prepared JSONL")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="miniGPT-test-ppo", help="wandb project name")
    
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
    
    # Initialize model configuration
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    
    # Check for existing checkpoint if resuming
    ckp_data = lm_checkpoint(
        lm_config, weight=args.save_weight, save_dir='checkpoints'
    ) if args.from_resume == 1 else None
    
    # ========== 3. Set up mixed precision training ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # No-op context for CPU, autocast for CUDA
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. Configure wandb logging ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wandb_module
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = (
            f"miniGPT-PPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LearningRate-{args.learning_rate}"
        )
        wandb_run = wandb_module.init(
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume
        )
        wandb = wandb_module  # use module for logging
        print(f"✅ Wandb initialized: {wandb_run.name} (ID: {wandb_run.id})")
    
    # ========== 5. Initialize models and data ==========
    # Base weight to initialize from (reasoning or full_sft)
    # base_weight = "reason" if args.reasoning == 1 else "full_sft"
    base_weight = "full_sft" # NOTE: Use the above when having an reasoning model
    
    # Actor model (policy being optimized)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    tokenizer.padding_side = 'left'  # PPO requires left-padding for generation
    
    # Old actor model (snapshot for importance sampling)
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # Reference model (frozen, for KL penalty)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Critic model (value function)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device) #type: ignore 
    
    # Reward model (external model for scoring)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # Training dataset and optimizers
    # Dataset preparation with mixer
    if args.data_config:
        from pathlib import Path
        from dataset.mixer import DatasetMixer
        
        if is_main_process():
            print(f"Using dataset config: {args.data_config}")
        
        mixer = DatasetMixer.from_yaml(args.data_config)
        validation = mixer.validate_mixture()
        
        if not validation['is_valid']:
            raise ValueError("Invalid mixture!")
        
        config_name = Path(args.data_config).stem
        train_jsonl = f"dataset/{mixer.config.phase}_{config_name}_train.jsonl"
        val_jsonl = f"dataset/{mixer.config.phase}_{config_name}_val.jsonl"
        
        if not args.use_prepared or not os.path.exists(train_jsonl):
            if is_main_process():
                mixer.prepare_dataset(train_jsonl, split="train")
                mixer.prepare_dataset(val_jsonl, split="validation")
            
            if dist.is_initialized():
                dist.barrier()
        
        args.data_path = train_jsonl
    
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # Count total iterations for learning rate schedulers
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    
    # Cosine annealing schedulers
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. Restore training state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap models with DistributedDataParallel ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization across GPUs
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore 
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore 
        
        # Wrap models for distributed training
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        
        # Move old actor to device (not wrapped as it's only used for inference)
        old_actor_model.to(args.device) #type: ignore 
    
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
            ppo_train_epoch(
                epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model,
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb
            )
        else:
            # Standard dataloader for normal training
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None), 
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            ppo_train_epoch(
                epoch, loader, len(loader), old_actor_model, ref_model,
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb
            )