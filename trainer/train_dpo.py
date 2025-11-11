import os
import sys

# Package path manipulation to ensure proper imports from parent directory
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Convert model logits to log probabilities for the given labels.
    
    This function computes per-token log probabilities by applying log_softmax
    to logits and then gathering the log probs corresponding to the actual
    tokens in the labels.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        labels: Target token indices of shape (batch_size, seq_len)
        
    Returns:
        Log probabilities of shape (batch_size, seq_len)
    """
    # Compute log softmax over vocabulary dimension
    log_probs = F.log_softmax(logits, dim=2)
    
    # Gather log probabilities for the actual tokens in the labels
    log_probs_per_token = torch.gather(
        log_probs, 
        dim=2, 
        index=labels.unsqueeze(2)
    ).squeeze(-1)
    
    return log_probs_per_token


def dpo_loss(
    ref_log_probs: torch.Tensor,
    policy_log_probs: torch.Tensor,
    mask: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Compute Direct Preference Optimization (DPO) loss.
    
    DPO optimizes a policy directly using preference data without reward modeling.
    It compares the policy model's log probabilities against a frozen reference
    model's log probabilities for both chosen and rejected responses.
    
    Key insight: The loss encourages the policy to increase the relative log
    probability of chosen responses over rejected ones compared to the reference model.
    
    Args:
        ref_log_probs: Log probabilities from reference model, shape (batch_size, seq_len)
        policy_log_probs: Log probabilities from policy model, shape (batch_size, seq_len)
        mask: Attention mask to ignore padding, shape (batch_size, seq_len)
        beta: Temperature parameter controlling deviation from reference model
        
    Returns:
        Mean DPO loss across the batch
    
    Implementation Notes:
        - Computes average log probs per sequence using mask to handle variable lengths
        - Splits batch into chosen and rejected halves
        - Computes log ratios for both policy and reference models
        - Uses log-sigmoid to encourage policy to favor chosen over rejected responses
    """
    # Calculate sequence lengths to normalize log probabilities
    # Add small epsilon to prevent division by zero for empty sequences
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    # Average log probabilities per sequence (accounting for padding)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # Split batch into chosen (preferred) and rejected responses
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # Compute log probability ratios for policy and reference
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # Compute final DPO loss: negative log-sigmoid of the difference
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()


def train_epoch(
    epoch: int,
    loader: DataLoader,
    iters: int,
    ref_model: torch.nn.Module,
    lm_config: MiniMindConfig,
    start_step: int = 0,
    wandb=None,
    beta: float = 0.1
) -> None:
    """
    Train model for one epoch using Direct Preference Optimization (DPO).
    
    DPO trains a policy model to generate responses that are preferred by humans
    without explicitly learning a reward model. It uses a frozen reference model
    to prevent the policy from deviating too far from its initial behavior.
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches of DPO data (chosen/rejected pairs)
        iters: Total number of iterations in this epoch
        ref_model: Frozen reference model (copy of initial policy)
        lm_config: Model configuration object
        start_step: Starting step number if resuming from checkpoint
        wandb: Weights & Biases logger instance (or None)
        beta: DPO temperature parameter controlling KL penalty strength
        
    Key DPO Characteristics:
        - Uses paired data: chosen (good) and rejected (bad) responses
        - Reference model remains frozen throughout training
        - Encourages policy to increase relative probability of chosen vs rejected
        - Beta parameter controls how much policy can deviate from reference
    """
    start_time = time.time()
    
    for step, batch in enumerate(loader, start=start_step + 1):
        # Load chosen/preferred and rejected responses
        x_chosen = batch['x_chosen'].to(args.device)  # Input tokens for chosen responses
        x_rejected = batch['x_rejected'].to(args.device)  # Input tokens for rejected responses
        y_chosen = batch['y_chosen'].to(args.device)  # Target tokens for chosen responses
        y_rejected = batch['y_rejected'].to(args.device)  # Target tokens for rejected responses
        mask_chosen = batch['mask_chosen'].to(args.device)  # Attention masks for chosen
        mask_rejected = batch['mask_rejected'].to(args.device)  # Attention masks for rejected
        
        # Concatenate chosen and rejected batches for parallel processing
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # Calculate learning rate based on current step (cosine schedule)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== Forward Passes ==========
        with autocast_ctx:
            # Reference model forward pass (frozen)
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            
            # Convert reference logits to log probabilities
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # Policy model forward pass (being trained)
            outputs = model(x)
            logits = outputs.logits
            
            # Convert policy logits to log probabilities
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # Compute DPO loss
            loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            
            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # ========== Backward Pass ==========
        scaler.scale(loss).backward()

        # ========== Optimizer Step ==========
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step and zero gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # ========== Logging & Monitoring ==========
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:'
            )
            
            if wandb: 
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # ========== Checkpointing ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # Determine filename suffix based on architecture
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract state dict (handle DDP wrapper)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # Save in half precision to reduce storage
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # Save full checkpoint with optimizer state
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, 
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, 
                wandb=wandb, save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main training script for MiniMind Direct Preference Optimization (DPO).
    
    DPO is a preference learning algorithm that directly optimizes a policy
    model using human preference data without explicit reward modeling. It
    maintains a frozen reference model to ensure the policy doesn't deviate
    too far from its initial behavior.
    
    The training data consists of pairs of responses (chosen and rejected)
    for the same prompt, where chosen responses are preferred by humans.
    """
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='dpo', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="Initial learning rate (suggest <=5e-8 to avoid catastrophic forgetting)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (steps)")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Maximum sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Data and initialization
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO training data path")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="Base weight for training")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect & resume training (0=no, 1=yes)")
    
    # DPO-specific parameters
    parser.add_argument('--beta', default=0.1, type=float, help="Beta parameter in DPO (KL penalty strength)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb project name")
    
    args = parser.parse_args()

    # ========== 1. Initialize environment and random seed ==========
    # Set up distributed training if available
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # Set random seed for reproducibility (different per process)
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
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize policy and reference models ==========
    # Initialize policy model (being trained)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'Policy model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # Initialize reference model (frozen copy of initial policy)
    # This prevents the policy from deviating too far from original behavior
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'Reference model parameters: {sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # Initialize dataset and training components
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore training state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, lm_config, start_step, wandb, args.beta)
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
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)