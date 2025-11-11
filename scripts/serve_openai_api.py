"""
OpenAI-Compatible API Server for MiniMind Models

Provides a FastAPI server that implements the OpenAI Chat Completions API format
for MiniMind language models. Supports both streaming and non-streaming responses,
LoRA adapters, and multiple model formats.

Features:
- OpenAI-compatible /v1/chat/completions endpoint
- Token-level streaming with Server-Sent Events (SSE)
- Support for native PyTorch and Transformers model formats
- LoRA (Low-Rank Adaptation) support for domain-specific models
- Configurable generation parameters (temperature, top_p, max_tokens)
- YaRN context length extrapolation support

Usage:
    # Start server with default settings (loads from ../model/)
    python scripts/serve_openai_api.py
    
    # Start with custom checkpoint
    python scripts/serve_openai_api.py \
        --load_from ../out \
        --weight full_sft \
        --hidden_size 768
    
    # Start with LoRA adapter
    python scripts/serve_openai_api.py --lora_weight lora_medical
    
API Example:
    curl -X POST http://127.0.0.1:8998/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "minimind", "messages": [{"role": "user", "content": "Hello!"}]}'

Requirements:
    - fastapi, uvicorn, pydantic
    - transformers, torch
    - MiniMind model files and tokenizer
"""

import argparse
import json
import os
import sys
import time
import warnings
from queue import Queue
from threading import Thread
from typing import List, Dict, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Add parent directory to path for model imports
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()

# Global variables for model and tokenizer (initialized in main)
model: Optional[MiniMindForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
device: Optional[torch.device] = None


def init_model(args: argparse.Namespace) -> tuple[MiniMindForCausalLM, AutoTokenizer]:
    """
    Initialize model and tokenizer based on command-line arguments.
    
    Supports two loading modes:
    1. Native PyTorch format: Loads from checkpoint in ../out/ directory
    2. Transformers format: Loads from HuggingFace format directory
    
    Optionally applies LoRA adapters for domain-specific fine-tuning.
    
    Args:
        args: Command-line arguments parsed by argparse
        
    Returns:
        Tuple of (model, tokenizer) loaded on specified device
    
    Notes:
        - Automatically detects CUDA or falls back to CPU
        - Prints total model parameters for verification
        - Sets model to evaluation mode and disables gradients
    """
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # Load model in either native PyTorch or Transformers format
    if 'model' in args.load_from:
        # Native PyTorch checkpoint loading
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # Initialize config and model
        model = MiniMindForCausalLM(
            MiniMindConfig(
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_hidden_layers,
                max_seq_len=args.max_seq_len,  # Note: renamed from max_position_embeddings in config
                use_moe=bool(args.use_moe),
                inference_rope_scaling=args.inference_rope_scaling
            )
        )
        
        # Load checkpoint weights
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        
        # Apply LoRA if specified
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # Transformers format loading
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    # Print model statistics
    param_count = sum(p.numel() for p in model.parameters())
    print(f'MiniMind model parameters: {param_count / 1e6:.2f}M')
    
    # Set to evaluation mode and move to device
    return model.eval().to(device), tokenizer


class ChatRequest(BaseModel):
    """
    Request model for /v1/chat/completions endpoint.
    
    Compatible with OpenAI API specification.
    """
    
    model: str
    messages: List[Dict[str, Any]]  # List of conversation turns
    temperature: float = 0.7  # Sampling temperature (0=deterministic, 1.0=creative)
    top_p: float = 0.92  # Nucleus sampling parameter
    max_tokens: int = 8192  # Maximum tokens to generate (not total context)
    stream: bool = False  # Enable token-by-token streaming
    tools: List[Any] = []  # Function calling tools (future feature)


class CustomStreamer(TextStreamer):
    """
    Custom text streamer for asynchronous token streaming.
    
    Wraps HuggingFace TextStreamer to push tokens to a queue for
    Server-Sent Events (SSE) streaming response.
    
    Args:
        tokenizer: Tokenizer for decoding token IDs to text
        queue: Queue for passing tokens to the streaming endpoint
    """
    
    def __init__(self, tokenizer: AutoTokenizer, queue: Queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Push decoded text to queue for streaming."""
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)  # Sentinel value to signal completion


def generate_stream_response(
    messages: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int
) -> Any:
    """
    Generate streaming response with Server-Sent Events (SSE).
    
    Runs generation in a separate thread and yields tokens as they are produced.
    
    Args:
        messages: Conversation history
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum new tokens to generate
    
    Yields:
        JSON-formatted SSE chunks compatible with OpenAI streaming format
    
    Notes:
        - Uses HuggingFace's generate() with custom streamer
        - Handles exceptions gracefully and returns error as JSON
        - Truncates prompt to max_tokens to prevent overflow
    """
    
    try:
        # Format messages with chat template and truncate
        new_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )[-max_tokens:]
        
        # Tokenize input
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
        
        # Setup streaming queue and streamer
        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)
        
        # Run generation in separate thread to avoid blocking
        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        
        Thread(target=_generate).start()
        
        # Stream tokens as they arrive
        while True:
            text = queue.get()
            if text is None:
                # End of generation
                yield json.dumps({
                    "choices": [{
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }, ensure_ascii=False)
                break
            
            # Yield token chunk
            yield json.dumps({
                "choices": [{"delta": {"content": text}}]
            }, ensure_ascii=False)
    
    except Exception as e:
        yield json.dumps({"error": str(e)}, ensure_ascii=False)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> Union[Dict[str, Any], StreamingResponse]:
    """
    OpenAI-compatible chat completions endpoint.
    
    Supports both streaming and non-streaming modes. In streaming mode,
    returns a StreamingResponse with Server-Sent Events. In non-streaming,
    returns a complete JSON response.
    
    Args:
        request: ChatRequest with messages and generation parameters
        
    Returns:
        StreamingResponse for streaming mode, or dict for standard mode
        
    Raises:
        HTTPException: If generation fails (returns 500 error)
    """
    
    try:
        if request.stream:
            # Streaming mode: return SSE response
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming mode: generate complete response
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                
                # Decode only the newly generated tokens
                answer = tokenizer.decode(
                    generated_ids[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
            
            # Return complete response in OpenAI format
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop"
                    }
                ]
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """
    Main entry point for API server.
    
    Parses command-line arguments, initializes model and tokenizer,
    and starts the FastAPI server with uvicorn.
    
    Command-line arguments:
        --load_from: Model loading path
        --save_dir: Model weights directory
        --weight: Weight name prefix
        --lora_weight: LoRA weight name (None to disable)
        --hidden_size: Hidden dimension size
        --num_hidden_layers: Number of hidden layers
        --max_seq_len: Maximum sequence length
        --use_moe: Enable MoE architecture (0 or 1)
        --inference_rope_scaling: Enable YaRN context extrapolation
        --device: Runtime device (cuda or cpu)
    """
    
    parser = argparse.ArgumentParser(description="MiniMind OpenAI-Compatible API Server")
    
    parser.add_argument(
        '--load_from', 
        default='../model', 
        type=str,
        help="Model loading path (model=native torch weights, other=transformers format)"
    )
    parser.add_argument(
        '--save_dir', 
        default='out', 
        type=str, 
        help="Model weights directory"
    )
    parser.add_argument(
        '--weight', 
        default='full_sft', 
        type=str, 
        help="Weight name prefix (pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo)"
    )
    parser.add_argument(
        '--lora_weight', 
        default='None', 
        type=str, 
        help="LoRA weight name (None=disabled, options: lora_identity, lora_medical)"
    )
    parser.add_argument(
        '--hidden_size', 
        default=512, 
        type=int, 
        help="Hidden dimension (512=Small-26M, 640=MoE-145M, 768=Base-104M)"
    )
    parser.add_argument(
        '--num_hidden_layers', 
        default=8, 
        type=int, 
        help="Number of hidden layers (Small/MoE=8, Base=16)"
    )
    parser.add_argument(
        '--max_seq_len', 
        default=8192, 
        type=int, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        '--use_moe', 
        default=0, 
        type=int, 
        choices=[0, 1], 
        help="Enable MoE architecture (0=no, 1=yes)"
    )
    parser.add_argument(
        '--inference_rope_scaling', 
        default=False, 
        action='store_true', 
        help="Enable RoPE position encoding extrapolation (4x for longer contexts)"
    )
    parser.add_argument(
        '--device', 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        type=str, 
        help="Runtime device"
    )
    
    args = parser.parse_args()
    
    # Set global device
    global device
    device = torch.device(args.device)
    
    # Initialize model and tokenizer
    global model, tokenizer
    model, tokenizer = init_model(args)
    
    # Start server
    print(f"\nStarting server on http://0.0.0.0:8998")
    print(f"Ready to serve requests...")
    uvicorn.run(app, host="0.0.0.0", port=8998)


if __name__ == "__main__":
    main()