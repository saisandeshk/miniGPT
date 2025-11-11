"""
Tokenizer Training Script for MiniMind

Trains a custom Byte-Level BPE (Byte Pair Encoding) tokenizer on a pre-training
corpus. The tokenizer uses a small vocabulary (6,400 tokens) to keep the model
size minimal, trading off some efficiency for reduced embedding layer parameters.

Key Features:
- Byte-Level BPE: Handles any Unicode text without out-of-vocabulary issues
- Custom special tokens: <|endoftext|>, <|im_start|>, <|im_end|> for ChatML format
- Jinja2 chat template: Supports tool calling, multi-turn conversations, and reasoning
- Small vocabulary: 6,400 tokens (vs 32K-150K in typical LLMs) for memory efficiency

Usage:
    python scripts/train_tokenizer.py
    
The trained tokenizer is saved in ../model/ directory and ready for immediate use
in training scripts.
"""

import random
import json
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
import os
from typing import Iterator, Dict, Any

random.seed(42)


def train_tokenizer() -> None:
    """
    Train a Byte-Level BPE tokenizer on pre-training data.
    
    Training Process:
    1. Load text samples from pretrain_hq.jsonl
    2. Initialize Byte-Level BPE tokenizer (handles any Unicode)
    3. Train with 6,400 vocab size on extracted text
    4. Setup special tokens for ChatML format
    5. Save tokenizer and manual config files
    
    The vocabulary size is intentionally small (6,400) to reduce model parameters,
    making it suitable for training tiny models on consumer hardware.
    
    Output Files:
        - ../model/tokenizer.json: Main tokenizer file
        - ../model/tokenizer_config.json: HuggingFace config with chat template
        - ../model/vocab.json & merges.txt: BPE vocabulary files
    """
    
    # Generator to read text samples from JSONL file
    def read_texts_from_jsonl(file_path: str) -> Iterator[str]:
        """
        Yield text field from each line in JSONL file.
        
        Args:
            file_path: Path to JSONL file with 'text' field
            
        Yields:
            Raw text string for each sample
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                yield data['text']

    data_path = '../dataset/pretrain_hq.jsonl'

    # Initialize Byte-Level BPE tokenizer
    # ByteLevel handles any Unicode text by operating on bytes
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) #type: ignore 

    # Define special tokens for ChatML format
    # These must match the indices in tokenizer_config.json
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # Setup BPE trainer with vocabulary size and special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # Ensure these tokens are included
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Read training text data
    texts = read_texts_from_jsonl(data_path)

    # Train tokenizer on the corpus
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Setup decoder to reverse tokenization
    tokenizer.decoder = decoders.ByteLevel() #type: ignore 

    # Verify special token indices are correctly assigned
    # These indices are critical for the model to recognize special tokens
    assert tokenizer.token_to_id("<|endoftext|>") == 0, "Pad/EOS token should be index 0"
    assert tokenizer.token_to_id("<|im_start|>") == 1, "BOS token should be index 1"
    assert tokenizer.token_to_id("<|im_end|>") == 2, "EOS token should be index 2"

    # Create output directory
    tokenizer_dir = "../model/"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Save tokenizer files
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("../model/")  # Saves vocab.json and merges.txt

    # Manually create tokenizer config with ChatML template
    # This config enables apply_chat_template() function in transformers
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        # Jinja2 chat template supporting tools, multi-turn, and reasoning
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\\\"name\\\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \\\"arguments\\\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    # Save configuration file
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved to ../model/")


def eval_tokenizer() -> None:
    """
    Evaluate and test the trained tokenizer.
    
    Performs the following checks:
    1. Loads the trained tokenizer
    2. Applies chat template to sample messages
    3. Verifies vocabulary size includes special tokens
    4. Encodes and decodes to check round-trip consistency
    
    This ensures the tokenizer is correctly configured before training.
    """
    
    from transformers import AutoTokenizer

    # Load the trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../model/")
    
    # Test chat template with sample conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant, always providing accurate responses!"},
        {"role": "user", "content": 'Where are you from?'},
        {"role": "assistant", "content": 'I am from Earth'}
    ]
    
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Formatted prompt:\n{new_prompt}\n")
    
    # Get actual vocabulary size (including special tokens)
    actual_vocab_size = len(tokenizer)
    print(f'Tokenizer actual vocabulary size: {actual_vocab_size}')
    
    # Test encoding
    model_inputs = tokenizer(new_prompt)
    print(f'Encoded input IDs length: {len(model_inputs["input_ids"])}')
    
    # Test decoding (round-trip)
    input_ids = model_inputs['input_ids']
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f'Decoded text matches original: {decoded_text == new_prompt}')


def main() -> None:
    """
    Main entry point for tokenizer training and evaluation.
    
    Runs both training and evaluation in sequence:
    1. Train new tokenizer on pre-training corpus
    2. Evaluate tokenizer on sample conversation
    """
    
    print("Starting tokenizer training...")
    train_tokenizer()
    
    print("\nEvaluating tokenizer...")
    eval_tokenizer()
    
    print("\nDone!")


if __name__ == '__main__':
    main()