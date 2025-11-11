"""
MiniMind Language Model Inference and Dialogue Script

This script provides a command-line interface for running inference with MiniMind models,
supporting both automated testing with predefined prompts and manual interactive dialogue.
Features include:
- Loading native PyTorch or HuggingFace transformers format models
- Optional LoRA weight loading
- Conversation history management
- Configurable generation parameters (temperature, top-p, etc.)
- Support for reasoning models with special templates
- Token streaming for real-time generation
"""

import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def init_model(args):
    """
    Initialize the language model and tokenizer based on command-line arguments.
    
    Supports two loading paths:
    1. Native MiniMind weights (when args.load_from == 'model')
    2. HuggingFace transformers format (when args.load_from is a path)
    
    Optionally applies and loads LoRA weights for efficient fine-tuning.
    
    Args:
        args: An argparse.Namespace containing the following attributes:
            - load_from (str): Model path ('model' for native weights, or transformers path)
            - save_dir (str): Directory containing model weights
            - weight (str): Weight filename prefix
            - lora_weight (str): LoRA weight name ('None' to disable)
            - hidden_size (int): Model hidden dimension
            - num_hidden_layers (int): Number of transformer layers
            - use_moe (int): Whether to use MoE architecture (0/1)
            - inference_rope_scaling (bool): Enable RoPE scaling for longer sequences
            - device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (model, tokenizer) where model is in evaluation mode on the specified device.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer


def main():
    """
    Main inference loop supporting both automated testing and manual dialogue modes.
    
    Provides two interaction modes:
    1. Auto Test: Cycles through predefined prompts for model evaluation
    2. Manual Input: Interactive chat with customizable conversation history
    
    Features:
    - Configurable conversation history length (must be even number)
    - Streaming generation for real-time responses
    - Special handling for reasoning models (enable_thinking flag)
    - Reproducible generation with fixed or random seeds
    
    Command-line arguments control model configuration, generation parameters,
    and runtime behavior.
    """
    parser = argparse.ArgumentParser(description="MiniMind Model Inference and Dialogue")
    parser.add_argument('--load_from', default='model', type=str, help="Model loading path ('model'=native torch weights, other path=transformers format)")
    parser.add_argument('--save_dir', default='out', type=str, help="Model weights directory")
    parser.add_argument('--weight', default='full_sft', type=str, help="Weight filename prefix (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA weight name (None means no LoRA, optional: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden dimension (512=Small-26M, 640=MoE-145M, 768=Base-104M)")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers (Small/MoE=8, Base=16)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Whether to use MoE architecture (0=No, 1=Yes)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Enable RoPE position encoding extrapolation (4x, only solves position encoding issues)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="Maximum generation length (note: not the model's actual long-context capability)")
    parser.add_argument('--temperature', default=0.85, type=float, help="Generation temperature, controls randomness (0-1, higher=more random)")
    parser.add_argument('--top_p', default=0.85, type=float, help="Nucleus sampling threshold (0-1)")
    parser.add_argument('--historys', default=0, type=int, help="Number of history conversation rounds to carry (must be even, 0=no history)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Device to run on")
    args = parser.parse_args()
    
    prompts = [
        'What are your specialties?',
        'Why is the sky blue?',
        'Please write a Python function to calculate Fibonacci sequence',
        'Explain the basic process of "photosynthesis"',
        'If it rains tomorrow, how should I go out?',
        'Compare the pros and cons of cats and dogs as pets',
        'Explain what machine learning is',
        'Recommend some Chinese food'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] Auto Test\n[1] Manual Input\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    for prompt in prompt_iter:
        setup_seed(2026)  # or setup_seed(random.randint(0, 2048))
        if input_mode == 0:
            print(f'👶: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = True  # Only for Reason model
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()