import os
import sys

# Package path manipulation to ensure proper imports from parent directory
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


def train_epoch(
    epoch: int, 
    loader: DataLoader, 
    iters: int, 
    lora_params: list, 
    start_step: int = 0, 
    wandb=None
) -> None:
    """
    Train the model for a single epoch using LoRA (Low-Rank Adaptation).
    
    This function implements LoRA fine-tuning where only the low-rank decomposition
    matrices (LoRA parameters) are trained while the base model remains frozen.
    This dramatically reduces memory usage and training time while maintaining
    model quality.
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches of training data
        iters: Total number of iterations in this epoch
        lora_params: List of LoRA parameters that require gradients
        start_step: Starting step number if resuming from checkpoint
        wandb: Weights & Biases logger instance (or None)
    
    Key LoRA Features:
        - Only LoRA parameters are optimized (base model is frozen)
        - Gradient clipping applied only to LoRA parameters
        - LoRA weights saved separately from base model
        - Memory-efficient fine-tuning with preserved base model weights
    """
    # Initialize loss function and timer
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    # Iterate through data batches, starting from the specified step
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # Move all tensors to the target device
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # Calculate learning rate based on current step (cosine schedule)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass with automatic mixed precision
        with autocast_ctx:
            res = model(X)
            
            # Compute per-token cross-entropy loss
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # Apply loss mask to ignore padding tokens and normalize
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # Add auxiliary loss (e.g., from MoE load balancing if enabled)
            loss += res.aux_loss
            
            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights only every accumulation_steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            
            # Apply gradient clipping only to LoRA parameters
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            # Optimizer step and zero gradients
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients and cache for memory efficiency
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Logging and monitoring
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # Scale back for reporting
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60  # Estimated time remaining
            
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '
                f'loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:'
            )
            
            if wandb: 
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # Model checkpointing (only on main process in distributed training)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # Save only LoRA weights (base model weights are frozen and unchanged)
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            save_lora(model, lora_save_path)
            
            # Save full checkpoint with optimizer state
            lm_checkpoint(
                lm_config, weight=args.lora_name, model=model, optimizer=optimizer, 
                scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main training script for MiniMind LoRA (Low-Rank Adaptation) Fine-tuning.
    
    This script performs parameter-efficient fine-tuning using LoRA, which trains
    only small rank decomposition matrices while keeping the base model frozen.
    This approach is memory-efficient and allows for multiple LoRA adapters to be
    trained for different tasks without modifying the base model.
    
    LoRA Advantages:
        - ~10-100x fewer trainable parameters
        - Preserves base model weights
        - Enables task-specific adapters
        - Lower memory usage and faster training
    
    Usage:
        Train domain-specific adapters (e.g., medical, legal) by setting lora_name
        and providing task-specific training data.
    """
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="Model save directory")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA weight name (e.g., lora_identity/lora_medical)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=1, help="Model saving interval (epochs)")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=512, type=int, help="Maximum sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Data and initialization
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA training data path")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="Base weight to train from (default: full_sft)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb project name")
    
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
        lm_config, weight=args.lora_name, save_dir='../checkpoints'
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
        wandb_run_name = (
            f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-"
            f"BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize model, apply LoRA, freeze non-LoRA parameters ==========
    # Initialize base model
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # Apply LoRA modifications to the model (inject low-rank matrices)
    apply_lora(model)
    
    # ========== Parameter Statistics ==========
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"Total LLM parameters: {total_params / 1e6:.3f} M")
    Logger(f"LoRA parameters: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA parameter ratio: {lora_params_count / total_params * 100:.2f}%")
    
    # ========== Freeze Non-LoRA Parameters ==========
    # Freeze base model weights and collect LoRA parameters for optimization
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # LoRA parameters are trainable
            param.requires_grad = True
            lora_params.append(param)
        else:
            # Base model parameters are frozen
            param.requires_grad = False
    
    # ========== 6. Initialize dataset and optimizer ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Optimizer only updates LoRA parameters (base model is frozen)
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. Restore training state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # Load LoRA weights only (strict=False allows missing base model weights)
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. Wrap model with DistributedDataParallel ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization across GPUs
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore 
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. Start main training loop ==========
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, lora_params, start_step, wandb)
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
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)