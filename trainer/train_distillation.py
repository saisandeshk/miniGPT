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
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = 'batchmean'
) -> torch.Tensor:
    """
    Compute KL divergence-based distillation loss between student and teacher logits.
    
    This implements the core knowledge distillation loss where the student learns
    to match the softened probability distribution of the teacher. The temperature
    parameter controls the softness of the distributions - higher temperature
    emphasizes learning from the teacher's overall distribution pattern rather
    than just the argmax.
    
    Args:
        student_logits: Raw logits from student model [..., vocab_size]
        teacher_logits: Raw logits from teacher model [..., vocab_size]
        temperature: Temperature for softening distributions (T>1 makes softer)
        reduction: Reduction method for KL divergence ('batchmean' recommended)
        
    Returns:
        Scaled KL divergence loss value
    """
    with torch.no_grad():
        # Compute teacher's softened probabilities and detach to prevent gradient flow
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # Compute student's softened log probabilities
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Calculate KL divergence (student || teacher)
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # Scale by temperature squared as per distillation best practices
    return (temperature ** 2) * kl


def train_epoch(
    epoch: int,
    loader: DataLoader,
    iters: int,
    teacher_model: torch.nn.Module,
    lm_config_student: MiniMindConfig,
    start_step: int = 0,
    wandb=None,
    alpha: float = 0.0,
    temperature: float = 1.0
) -> None:
    """
    Train student model for one epoch with knowledge distillation.
    
    This implements the main distillation training loop combining two losses:
    1. Cross-entropy loss against ground truth labels
    2. KL divergence loss against teacher model's softened distributions
    
    The total loss is a weighted combination: loss = alpha * CE + (1-alpha) * Distill
    
    Args:
        epoch: Current epoch number (0-indexed)
        loader: DataLoader providing training batches
        iters: Total iterations in this epoch
        teacher_model: Frozen teacher model for distillation
        lm_config_student: Student model configuration
        start_step: Starting step if resuming from checkpoint
        wandb: Experiment tracking logger (or None)
        alpha: Weight for CE loss (0=full distillation, 1=standard training)
        temperature: Temperature for softening distributions in distillation
    
    Training Features:
        - Mixed precision training for memory efficiency
        - Gradient accumulation for large effective batch sizes
        - Gradient clipping for stability
        - Teacher model remains in eval mode and frozen
        - Separate logging of CE and distillation components
    """
    start_time = time.time()
    
    # Prepare teacher model: set to eval mode and freeze all parameters
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    # Main training loop
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # Move data to target device
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # Calculate learning rate based on current step (cosine schedule)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ========== Forward Passes ==========
        # Student model forward pass with mixed precision
        with autocast_ctx:
            res = model(X)
            student_logits = res.logits

        # Teacher model forward pass (only in eval mode and no_grad context)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # Handle potential vocabulary size mismatch by truncating
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ========== Loss Computation ==========
        # 1) Cross-Entropy Loss against ground truth
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            Y.view(-1),
            ignore_index=0,  # Ignore padding token
            reduction='none'
        )
        # Apply loss mask and normalize
        ce_loss = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        
        # Add MoE auxiliary loss if applicable
        if lm_config_student.use_moe:
            ce_loss += res.aux_loss

        # 2) Distillation Loss against teacher
        if teacher_model is not None:
            # Only compute distillation loss on non-padded positions
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) Combined Loss = alpha * CE + (1-alpha) * Distill
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # ========== Backward Pass & Optimization ==========
        scaler.scale(loss).backward()

        # Update weights only every accumulation_steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale gradients for clipping
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
                f'loss:{current_loss:.6f} ce:{ce_loss.item():.4f} '
                f'distill:{distill_loss.item():.4f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:'
            )
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": ce_loss.item(),
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "lr": current_lr,
                    "epoch_Time": eta_min
                })

        # ========== Checkpointing ==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            
            # Determine filename suffix based on architecture
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            
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
                lm_config_student, weight=args.save_weight, model=model, 
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, 
                wandb=wandb, save_dir='../checkpoints'
            )
            model.train()


if __name__ == "__main__":
    """
    Main training script for MiniMind Knowledge Distillation.
    
    This script performs knowledge distillation from a larger teacher model to a
    smaller student model. It combines ground truth supervision with teacher
    supervision using a weighted loss function. Supports distributed training,
    mixed precision, and checkpoint resumption.
    """
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    
    # Directory and naming
    parser.add_argument("--save_dir", type=str, default="../out", help="Model save directory")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="Prefix for saved weights")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    # Logging and saving intervals
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval (steps)")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval (steps)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for training")
    
    # Data and paths
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="Training data path")
    
    # Student model architecture
    parser.add_argument('--student_hidden_size', default=512, type=int, help="Student model hidden dimension")
    parser.add_argument('--student_num_layers', default=8, type=int, help="Student model number of layers")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture (0=no, 1=yes)")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="Base weight for student model")
    
    # Teacher model architecture
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="Teacher model hidden dimension")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="Teacher model number of layers")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="Base weight for teacher model")
    
    # Distillation parameters
    parser.add_argument('--alpha', default=0.5, type=float, help="CE loss weight: total_loss = alpha*CE + (1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="Distillation temperature (recommended: 1.0-2.0)")
    
    # Resume training
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect and resume training (0=no, 1=yes)")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb project name")
    
    args = parser.parse_args()

    # ========== 1. Initialize distributed environment and random seed ==========
    # Set up distributed training if available
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # Set random seed for reproducibility (different per process)
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Configure directories, model configs, check checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize student and teacher configurations
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.use_moe)
    )
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.use_moe)
    )
    
    # Check for existing checkpoint if resuming
    ckp_data = lm_checkpoint(
        lm_config_student, weight=args.save_weight, save_dir='../checkpoints'
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
        wandb_run_name = (
            f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-"
            f"Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        )
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume) #type: ignore 
    
    # ========== 5. Initialize student and teacher models ==========
    # Initialize student model
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'Student model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # Initialize teacher model (frozen)
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    Logger(f'Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    
    # Initialize dataset and optimizer
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore training state from checkpoint ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. Wrap student model with DistributedDataParallel ==========
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
            train_epoch(
                epoch, loader, len(loader) + start_step + 1, teacher_model,
                lm_config_student, start_step, wandb, args.alpha, args.temperature
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
            train_epoch(
                epoch, loader, len(loader), teacher_model,
                lm_config_student, 0, wandb, args.alpha, args.temperature
            )