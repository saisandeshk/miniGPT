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


def calculate_rewards(
    prompts: list[str], 
    responses: list[str], 
    reward_model, 
    reward_tokenizer
) -> torch.Tensor:
    """
    Compute comprehensive reward scores for generated responses.
    
    This function integrates multiple reward signals:
    1. **Format Rewards**: For reasoning models, checks if responses follow the
       correct `<think>...</think><answer>...</answer>` structure
    2. **Tag Counting Rewards**: Validates proper usage of reasoning tags
    3. **Reward Model Scores**: Uses an external reward model to evaluate response quality
    
    Args:
        prompts: List of input prompts (batch_size)
        responses: List of generated responses (batch_size * num_generations)
        reward_model: External reward model for scoring response quality
        reward_tokenizer: Tokenizer for the reward model
        
    Returns:
        Tensor of reward scores of shape (batch_size * num_generations,)
    
    Reward Calculation Details:
        - Format reward: 0.5 for correct structure, 0.0 otherwise
        - Tag counting: 0.25 for each correctly used tag (<think>, </think>, <answer>, </answer>)
        - Reward model: Scores clipped to [-scale, scale] range
        - For reasoning models: Combines full response score (40%) with extracted answer score (60%)
    """
    def reasoning_model_reward(rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute format-based rewards for reasoning model responses.
        
        Validates that responses follow the expected `<think>...</think><answer>...</answer>` format
        and counts correct usage of reasoning tags.
        """
        # Pattern 1: Exact format without extra newlines between tags
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        # Pattern 2: Format with an extra newline between think and answer sections
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        # Check if responses match either pattern
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # Assign format rewards: 0.5 for correct structure, 0.0 otherwise
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
        batch_size = len(prompts)
        scale = 3.0  # Clipping range for reward scores

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # Parse prompt into role-based message format
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # Score the full response
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)  # Clip to [-scale, scale]

                # For reasoning models: also score the extracted answer content separately
                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
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


def grpo_train_epoch(
    epoch: int,
    loader: DataLoader,
    iters: int,
    ref_model: MiniMindForCausalLM,
    reward_model,
    reward_tokenizer,
    start_step: int = 0,
    wandb=None
) -> None:
    """
    Train model for one epoch using Group Relative Policy Optimization (GRPO).
    
    GRPO is a reinforcement learning algorithm that improves language models by:
    1. Generating multiple responses per prompt (num_generations)
    2. Computing rewards for each response
    3. Normalizing rewards within each prompt's group (group-relative)
    4. Computing per-token advantages and policy loss with KL penalty
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing RL training data
        iters: Total number of iterations in this epoch
        ref_model: Frozen reference model for KL penalty
        reward_model: External reward model for scoring responses
        reward_tokenizer: Tokenizer for the reward model
        start_step: Starting step if resuming from checkpoint
        wandb: Experiment tracking logger (or None)
    
    Key GRPO Features:
        - Group-relative advantage estimation (no value function needed)
        - KL divergence penalty to prevent policy collapse
        - Per-token policy gradient updates
        - Multiple generations per prompt for better exploration
        - Mixed precision training support
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
            
            # Generate multiple completions per prompt
            outputs = model_for_gen.generate(
                **prompt_inputs, 
                max_new_tokens=args.max_gen_len, 
                do_sample=True, 
                temperature=0.8,
                num_return_sequences=args.num_generations, 
                pad_token_id=tokenizer.pad_token_id
            )  # [B*num_gen, P+R]
        
        # Extract completion tokens (remove prompt)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]

        # ========== Log Probability Calculation ==========
        def get_per_token_logps(mdl, input_ids: torch.Tensor, n_keep: int) -> torch.Tensor:
            """
            Compute per-token log probabilities for the last n_keep tokens.
            
            Uses logits_to_keep parameter for efficiency when only the last
            tokens' logits are needed for loss computation.
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
        outputs = cast(torch.Tensor, outputs) # NOTE: Verify this! 
        per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # ========== Reward Calculation ==========
        # Decode completions to text for reward calculation
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Calculate rewards using external reward model
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # ========== Group-Advantage Estimation ==========
        # Reshape rewards into groups by prompt
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        
        # Compute mean and std for each group, then repeat for each generation
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        
        # Compute normalized advantages (clipped for stability)
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        
        # Global advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # ========== Completion Mask Creation ==========
        # Create mask to ignore tokens after EOS
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # Binary mask: 1 for tokens before/inclusive of EOS, 0 after
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= 
            eos_idx.unsqueeze(1)
        ).int()  # [B*num_gen, R]

        # ========== Policy Loss Computation ==========
        # KL divergence between policy and reference
        kl_div = ref_per_token_logps - per_token_logps
        
        # Per-token KL penalty (r - log(r) - 1 where r = p_ref/p_policy)
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        
        # GRPO loss: advantage-weighted policy improvement with KL penalty
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - 
            args.beta * per_token_kl
        )  # [B*num_gen, R]
        
        # Average loss over non-padded tokens, then over batch
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / 
            completion_mask.sum(dim=1)
        ).mean() / args.accumulation_steps  # scalar

        # ========== Backward Pass ==========
        loss.backward()

        # ========== Optimizer Step ==========
        if (step + 1) % args.accumulation_steps == 0:
            # Gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step and learning rate scheduler update
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # ========== Logging & Monitoring ==========
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(
                f'Epoch: {epoch+1}, Step: {step}/{iters}, '
                f'Actor Loss: {policy_loss_val:.6f}, Reward: {avg_reward_val:.6f}, '
                f'Avg Response Len: {avg_len_val:.2f}, LR: {current_lr:.2e}'
            )

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
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
        # Explicitly delete tensors and collect garbage due to memory-intensive nature
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    """
    Main training script for MiniMind Group Relative Policy Optimization (GRPO).
    
    GRPO is a reinforcement learning algorithm for aligning language models with
    human preferences. It uses a group of responses per prompt to compute relative
    advantages and updates the policy with a KL penalty to a frozen reference model.
    
    This implementation includes special support for reasoning models that generate
    structured responses with <think>...</think><answer>...</answer> format.
    
    Key Components:
        - Policy model: The model being optimized
        - Reference model: Frozen copy of initial policy for KL penalty
        - Reward model: External model (internlm2-1_8b-reward) for scoring responses
    
    Training Process:
        1. Generate multiple responses per prompt (num_generations)
        2. Calculate rewards using format validation and reward model
        3. Compute group-relative advantages
        4. Update policy with clipped objective and KL penalty
        5. Log metrics and save checkpoints
    """
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='grpo', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (number of prompts per batch)")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
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
    parser.add_argument("--num_generations", type=int, default=8, help="Number of responses to generate per prompt")
    
    # RL parameters
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='Model type (0=regular, 1=reasoning)')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward model path")
    
    # Data and resumption
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF training data path")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb project name")
    
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
    # Note: Uses swanlab as a wandb-compatible experiment tracker
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize models, data, and optimizer ==========
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
    
    # ========== 7. Wrap policy model with DistributedDataParallel ==========
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
            grpo_train_epoch(
                epoch, loader, len(loader) + start_step + 1, ref_model, 
                reward_model, reward_tokenizer, start_step, wandb
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
            grpo_train_epoch(
                epoch, loader, len(loader), ref_model, 
                reward_model, reward_tokenizer, 0, wandb
            )