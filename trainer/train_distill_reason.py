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
    tokenizer, 
    lm_config: MiniMindConfig,
    start_step: int = 0, 
    wandb=None
) -> None:
    """
    Train the model for a single epoch with reasoning distillation loss.
    
    This function implements the core training loop with special emphasis on
    reasoning tags (<think>, </think>, <answer>, </answer>) that receive 
    higher loss weights during distillation.
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches of training data
        iters: Total number of iterations in this epoch
        tokenizer: Tokenizer for encoding/decoding text
        lm_config: Model configuration object
        start_step: Starting step number if resuming from checkpoint
        wandb: Weights & Biases logger instance (or None)
        
    Key Features:
        - Special weighting (10x) for reasoning tags to emphasize reasoning patterns
        - Mixed precision training with gradient scaling
        - Gradient accumulation for large effective batch sizes
        - Gradient clipping for training stability
        - Distributed training support
        - Periodic checkpointing and logging
    """
    # Token IDs for reasoning tags - these will receive special loss weighting
    # <think> and </think> mark the reasoning process
    # <answer> and </answer> mark the final answer
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    
    # Loss function with no reduction - we'll apply custom weighting
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
            
            # Calculate per-token loss
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # Identify positions of special reasoning tags and increase their weights
            # This is crucial for reasoning distillation - we want the model to
            # pay special attention to learning when to start/end thinking
            sp_ids = torch.isin(Y.view(-1),
                                torch.tensor(start_of_think_ids + end_of_think_ids
                                             + start_of_answer_ids + end_of_answer_ids
                                             ).to(args.device))
            
            # Apply custom loss weighting: 10x weight for reasoning tags
            loss_mask = loss_mask.view(-1)
            loss_mask_sum = loss_mask.sum()
            loss_mask[sp_ids] = 10  # Increase weight for thought tags by 10x
            
            # Apply loss mask and normalize
            loss_mask = loss_mask.view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask_sum
            
            # Add auxiliary loss (e.g., from MoE load balancing)
            loss += res.aux_loss
            
            # Scale loss for gradient accumulation
            loss = loss / args.accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update weights only every accumulation_steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # Optimizer step and zero gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Logging and monitoring
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')
            
            if wandb: 
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # Model checkpointing (only on main process in distributed training)
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
                optimizer=optimizer, scaler=scaler, epoch=epoch, 
                step=step, wandb=wandb, save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main training script for MiniMind Reasoning Distillation.
    
    This script trains a MiniMind model to perform reasoning through distillation,
    with special emphasis on learning to generate <think>...</think> sections
    before providing answers. It supports distributed training, mixed precision,
    gradient accumulation, and checkpoint resumption.
    """
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    
    # Directory and naming arguments
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='reason', type=str, help="Prefix name for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (steps)")
    
    # Model architecture parameters
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Maximum sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Whether to use MoE architecture (0=no, 1=yes)")
    
    # Data and weight initialization
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="Reasoning distillation data path")
    parser.add_argument('--from_weight', default='dpo', type=str, help="Base weight to fine-tune from (default: dpo)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Weights & Biases integration
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb project name")
    
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
    # Note: This imports swanlab as wandb, which is a Chinese experiment tracking platform
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb  # SwanLab is a Chinese alternative to wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize model, dataset, and optimizer ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # Set up distributed sampler for multi-GPU training
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore training state from checkpoint if resuming ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap model with DistributedDataParallel ==========
    if dist.is_initialized():
        # Ignore certain buffers that don't need synchronization
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} #type: ignore
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Start training loop ==========
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
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
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps, starting from step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + start_step + 1, tokenizer, lm_config, start_step, wandb)
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
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)