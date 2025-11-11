"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

This module provides a minimal, self-contained LoRA implementation that injects
trainable low-rank matrices into a pre-trained model's linear layers. Only the
LoRA parameters are updated during fine-tuning, keeping the original model frozen.

Key features:
- Injects LoRA into square linear layers (e.g., attention projections, FFNs)
- Gaussian initialization for matrix A, zeros for matrix B
- Helper functions for loading/saving only LoRA weights
- Memory-efficient fine-tuning with minimal code complexity
"""

import torch
from torch import optim, nn
from typing import Optional


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module for injecting trainable parameters.
    
    LoRA approximates weight updates with low-rank decomposition:
    ΔW = B * A, where A ∈ ℝ^(r×d) and B ∈ ℝ^(d×r)
    
    During forward pass: output = W*x + B*A*x
    where W is frozen and B*A are trainable.
    
    Args:
        in_features: Input dimension (must match original linear layer)
        out_features: Output dimension (must match original linear layer)
        rank: LoRA rank (r). Lower rank = fewer parameters but less capacity.
              Typical values: 4, 8, 16, 32, 64
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        
        # Low-rank matrix A: projects input to rank dimension
        # Initialized with Gaussian to break symmetry
        self.A = nn.Linear(in_features, rank, bias=False)
        
        # Low-rank matrix B: projects from rank back to output dimension
        # Initialized to zeros so LoRA initially contributes nothing
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights
        self.A.weight.data.normal_(mean=0.0, std=0.02)  # Gaussian initialization
        self.B.weight.data.zero_()  # Zero initialization (critical!)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layers.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return self.B(self.A(x))


def apply_lora(model: nn.Module, rank: int = 8) -> None:
    """
    Inject LoRA modules into all square linear layers of a model.
    
    This function modifies the model in-place by:
    1. Finding all nn.Linear layers with square weight matrices
    2. Creating a LoRA module for each
    3. Wrapping the original forward pass to include LoRA output
    
    Why square matrices? LoRA is typically applied to:
    - Attention Q/K/V/O projections
    - Feed-forward network layers
    - Skip non-square layers like lm_head (vocab_size != hidden_dim)
    
    Args:
        model: PyTorch model to inject LoRA into
        rank: LoRA rank for all injected modules
    
    Example:
        >>> model = MiniMindModel()
        >>> apply_lora(model, rank=16)
        >>> # Now only LoRA parameters are trainable
        >>> trainable_params = [p for p in model.parameters() if p.requires_grad]
    """
    
    for name, module in model.named_modules():
        # Only target square linear layers (same in/out features)
        # This avoids applying LoRA to embedding or lm_head layers
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # Create LoRA module and move to same device as parent model
            # Determine device from model parameters (fallback to CPU if none)
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(device)
            setattr(module, "lora", lora)
            
            # Store original forward method
            original_forward = module.forward
            
            # Create new forward method that adds LoRA output
            # Uses default arguments to bind values in closure (avoids late-binding bug)
            def forward_with_lora(input: torch.Tensor, original_layer=original_forward, lora_layer=lora) -> torch.Tensor:
                return original_layer(input) + lora_layer(input)
            
            module.forward = forward_with_lora



def load_lora(model: nn.Module, path: str) -> None:
    """
    Load LoRA weights from a checkpoint file.
    
    Only loads parameters that match `*.lora.*` pattern, leaving
    the base model weights untouched. This is memory-efficient as
    LoRA weights are typically <1% of base model size.
    
    Args:
        model: Model with LoRA modules already applied
        path: Path to LoRA checkpoint file (.pth or .pt)
    
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If LoRA modules don't exist in model or checkpoint
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    state_dict = torch.load(path, map_location=device)
    
    # Extract only LoRA parameters for each module
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # Filter state_dict to get only this module's LoRA weights
            prefix = f'{name}.lora.'
            lora_state = {
                k.replace(prefix, ''): v 
                for k, v in state_dict.items() 
                if k.startswith(prefix)
            }
            module.lora.load_state_dict(lora_state) #type: ignore


def save_lora(model: nn.Module, path: str) -> None:
    """
    Save only LoRA parameters to a checkpoint file.
    
    This creates a small checkpoint containing only trainable LoRA weights,
    not the full model. For distribution and storage efficiency.
    
    Args:
        model: Model with LoRA modules
        path: Output path for checkpoint file
    
    Example:
        >>> save_lora(model, 'lora_checkpoint.pth')
        >>> # File size will be ~MBs instead of GBs
    """
    
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # Prefix keys with module path for easy loading
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()} #type: ignore 
            state_dict.update(lora_state)
    
    torch.save(state_dict, path)