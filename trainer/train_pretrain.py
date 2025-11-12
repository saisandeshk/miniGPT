import os
import sys
from pathlib import Path

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
from dataset.lm_dataset import PretrainDataset
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
    start_step: int = 0, 
    wandb=None
) -> None:
    """
    Train the model for a single epoch during the pretraining phase.
    
    This function implements the core pretraining loop where the model learns
    to predict the next token in a sequence from a large corpus of text.
    Pretraining is the first stage in training a language model, where it learns
    general language patterns before task-specific fine-tuning.
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing batches of pretraining data
        iters: Total number of iterations in this epoch
        start_step: Starting step number if resuming from checkpoint
        wandb: Weights & Biases logger instance (or None)
    
    Pretraining Characteristics:
        - Uses causal language modeling objective (next-token prediction)
        - Trains on large-scale unstructured text data
        - Learns general language representations
        - Foundation for all downstream fine-tuning tasks
    
    Key Training Features:
        - Cross-entropy loss with masked padding tokens
        - Mixed precision training for memory efficiency
        - Gradient accumulation for large effective batch sizes
        - Gradient clipping for training stability
        - Distributed training support across multiple GPUs
        - Periodic checkpointing and logging
    """
    # Initialize loss function and timer
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    # Iterate through data batches, starting from the specified step
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # Move all tensors to the target device (GPU/CPU)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Optimizer step and zero gradients
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients and cache for memory efficiency
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Logging and monitoring
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # Scale back for accurate reporting
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
            
            # Determine filename suffix based on architecture
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # Extract state dict (handle DDP wrapper)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # Save in half precision to reduce storage requirements
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # Save full checkpoint with optimizer state for training resumption
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main pretraining script for MiniMind Language Model.
    
    This script performs causal language model pretraining on a large text corpus.
    Pretraining is the foundational stage where the model learns general language
    patterns and world knowledge before task-specific fine-tuning.
    
    Training Pipeline:
        1. Initialize model from scratch or resume from checkpoint
        2. Load large-scale pretraining dataset
        3. Train with next-token prediction objective
        4. Save periodic checkpoints and logs
    
    Typical Usage:
        # Train from scratch for 1-2 epochs
        python train_pretrain.py --epochs 1 --learning_rate 5e-4
        
        # Resume training from checkpoint
        python train_pretrain.py --from_resume 1 --epochs 3
    
    Architecture Support:
        - Standard dense transformer
        - Mixture of Experts (MoE) architecture (set use_moe=1)
    """
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (recommend 1 for zero-shot or 2-6 for full training)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (steps)")
    
    # Model architecture
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden layer dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=512, type=int, help="Maximum sequence length for training")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    
    # Data and initialization
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="Pretraining data path (JSONL file)")
    parser.add_argument("--data_config", type=str, default=None, help="Path to dataset mixture YAML config (alternative to --data_path)")
    parser.add_argument("--use_prepared", action="store_true", help="Use pre-prepared JSONL from data_config (skip re-preparation)")
    parser.add_argument('--from_weight', default='none', type=str, help="Base weight for training, 'none' means train from scratch")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect & resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb project name")
    
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
    
    # Check for existing checkpoint if resuming training
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
        wandb_run_name = (
            f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-"
            f"LearningRate-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize model, dataset, and optimizer ==========
    # Initialize model and tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # Initialize pretraining dataset (large-scale text corpus)
    # Support two modes:
    # 1. Direct JSONL file (--data_path)
    # 2. Dataset mixture config (--data_config) with automatic preparation
    if args.data_config:
        from dataset.mixer import DatasetMixer
        
        if is_main_process():
            print(f"Using dataset mixture config: {args.data_config}")
        
        # Load mixer configuration
        mixer = DatasetMixer.from_yaml(args.data_config)
        
        # Validate mixture ratios
        validation = mixer.validate_mixture()
        if is_main_process():
            print(f"Mixture validation: {validation}")
        
        if not validation['is_valid']:
            raise ValueError("Invalid mixture ratios! They must sum to 1.0")
        
        # Generate output filenames based on config
        config_name = Path(args.data_config).stem  # e.g., "default"
        train_jsonl = f"../dataset/{mixer.config.phase}_{config_name}_train.jsonl"
        
        # Prepare dataset (or skip if already prepared)
        if not args.use_prepared or not os.path.exists(train_jsonl):
            if is_main_process():
                print("Preparing dataset from mixture config...")
                train_jsonl = mixer.prepare_dataset(
                    output_file=train_jsonl,
                    split="train"
                )
            
            # Wait for main process to finish preparation
            if dist.is_initialized():
                dist.barrier()
        else:
            if is_main_process():
                print(f"Using pre-prepared dataset: {train_jsonl}")
        
        # Load the prepared JSONL file
        train_ds = PretrainDataset(train_jsonl, tokenizer, max_length=args.max_seq_len) #type: ignore
        
        if is_main_process():
            print(f"Loaded {len(train_ds)} training samples from {train_jsonl}")
    else:
        # Original mode: direct JSONL file
        train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len) #type: ignore
        if is_main_process():
            print(f"Loaded {len(train_ds)} training samples from {args.data_path}")
    
    # Set up distributed sampler for multi-GPU training
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Initialize AdamW optimizer
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
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
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
            train_epoch(epoch, loader, len(loader), 0, wandb)