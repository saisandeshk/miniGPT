"""
Model Format Conversion Utilities

Converts between native PyTorch checkpoints and HuggingFace Transformers format.
Supports both MiniMind's custom architecture and Llama-compatible format for
broader ecosystem compatibility.

Usage:
    python scripts/convert_model.py
"""

import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_torch2transformers_minimind(torch_path: str, transformers_path: str, dtype: torch.dtype = torch.float16) -> None:
    """
    Convert native PyTorch checkpoint to MiniMind Transformers format.
    
    Saves model with MiniMindConfig for use with AutoModelForCausalLM.
    Preserves full model architecture and native compatibility.
    
    Args:
        torch_path: Path to native PyTorch checkpoint (.pth file)
        transformers_path: Output directory for Transformers format
        dtype: Precision to save model (float16 recommended for size)
    """
    
    # Register MiniMind classes for AutoModel detection
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    
    # Initialize model and load weights
    lm_model = MiniMindForCausalLM(lm_config)
    lm_model.load_state_dict(state_dict, strict=False)
    # Convert model precision
    lm_model = lm_model.to(dtype) #type: ignore   
        
    # Print model size
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'Model parameters: {model_params / 1e6:.2f}M = {model_params / 1e9:.3f}B (Billion)')
    
    # Save in Transformers format
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    
    print(f"Model saved in MiniMind-Transformers format: {transformers_path}")


def convert_torch2transformers_llama(torch_path: str, transformers_path: str, dtype: torch.dtype = torch.float16) -> None:
    """
    Convert native PyTorch checkpoint to Llama-compatible Transformers format.
    
    Creates LlamaConfig and LlamaForCausalLM for maximum third-party compatibility
    with tools like llama.cpp, vLLM, etc. Architecture remains identical.
    
    Args:
        torch_path: Path to native PyTorch checkpoint (.pth file)
        transformers_path: Output directory for Transformers format
        dtype: Precision to save model (float16 recommended for size)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    
    # Create Llama-compatible config
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    
    # Initialize and load Llama model
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    # Convert model precision
    llama_model = llama_model.to(dtype)  #type: ignore  
    
    # Save in Transformers format
    llama_model.save_pretrained(transformers_path)
    
    # Print model size
    model_params = sum(p.numel() for p in llama_model.parameters() if p.requires_grad)
    print(f'Model parameters: {model_params / 1e6:.2f}M = {model_params / 1e9:.3f}B (Billion)')
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    
    print(f"Model saved in Llama-Transformers format: {transformers_path}")


def convert_transformers2torch(transformers_path: str, torch_path: str) -> None:
    """
    Convert Transformers format back to native PyTorch checkpoint.
    
    Useful for loading models saved in Transformers format back into
    native training scripts.
    
    Args:
        transformers_path: Path to Transformers model directory
        torch_path: Output path for PyTorch checkpoint (.pth file)
    """
    
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save(model.state_dict(), torch_path)
    print(f"Model saved in PyTorch format: {torch_path}")


if __name__ == '__main__':
    # Configuration for conversion
    # Edit these values as needed for your model
    lm_config = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=16,
        max_seq_len=8192,
        use_moe=False
    )

    # Paths
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../MiniMind2'

    # Perform conversion (Llama format for ecosystem compatibility)
    convert_torch2transformers_llama(torch_path, transformers_path)

    # Alternative: Convert Transformers back to PyTorch format
    # convert_transformers2torch(transformers_path, torch_path)