"""
Dataset classes for MiniGPT training pipeline.

This module provides PyTorch Dataset implementations for different stages of
language model training:
- PretrainDataset: For causal language modeling (next-token prediction)
- SFTDataset: For supervised fine-tuning on conversations
- DPODataset: For Direct Preference Optimization
- RLAIFDataset: For Reinforcement Learning from AI Feedback

Each dataset handles tokenization, sequence padding, and loss masking 
appropriately for its specific training objective.
"""

import json
import os
from typing import Dict, List, Any, Tuple, cast

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    Dataset for pre-training stage (causal language modeling).
    
    Loads raw text data and tokenizes it for next-token prediction.
    The model learns to predict each token based on all previous tokens.
    
    Args:
        data_path: Path to JSONL file with 'text' field
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load and parse JSONL file into list of samples."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            X: Input token IDs (all tokens except last)
            Y: Target token IDs (all tokens except first)
            loss_mask: Boolean mask indicating which tokens to compute loss on
                      (1 for real tokens, 0 for padding)
        """
        sample = self.samples[index]

        # Tokenize the raw text
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # Create input-target pairs for causal LM
        X = torch.tensor(input_ids[:-1], dtype=torch.long) # TODO: Check this! - Error: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
        Y = torch.tensor(input_ids[1:], dtype=torch.long) # TODO: Check this! 
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long) # TODO: Check this! 
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) on conversational data.
    
    Processes multi-turn conversations and creates loss masks that only
    compute loss on assistant responses (not on user inputs or system messages).
    
    Args:
        jsonl_path: Path to JSONL file with 'conversations' field
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        
        # Precompute special token IDs for assistant start/end markers
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load conversation data from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations: List[Dict[str, Any]]) -> str: 
        """
        Build a ChatML-formatted conversation string.
        
        Args:
            conversations: List of conversation turns with role/content
            
        Returns:
            Formatted prompt string with special tokens
        """
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and 
                                                   conversations[0]["role"] == "system" and 
                                                   conversations[0].get("functions")) else None
        return cast(str, self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        ))

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        """
        Generate loss mask that only computes loss on assistant responses.
        
        The mask identifies assistant responses by finding 'assistant' start tokens
        and marks all tokens until the end-of-sequence token for loss calculation.
        
        Args:
            input_ids: Tokenized conversation IDs
            
        Returns:
            Binary mask (1=compute loss, 0=skip loss)
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # Check if current position marks start of assistant response
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                
                # Find end of assistant response
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # Mark tokens in assistant response for loss computation
                # Start from +2 to skip the 'assistant' token itself
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                    
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single SFT training sample."""
        sample = self.samples[index]
        
        # Build conversation prompt with chat template
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # Generate dynamic loss mask for assistant responses
        loss_mask = self._generate_loss_mask(input_ids)

        # Create training data pairs
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Align mask with predictions
        
        # Debug helper: Print token-by-token mask status
        # print(f"\n--- Sample {index} Token Loss Mask (length: {len(input_ids)}) ---")
        # for i, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        #     token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
        #     token_str = token_str.replace('\n', '\\n').replace('\t', '\\t')  # Handle invisible chars
        #     print(f"Token {i:3d}: {token_id:5d} -> '{token_str:10s}' | mask: {mask}")
        # print(f"--- End of Sample {index} ---")
        
        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimization (DPO).
    
    Loads pairs of chosen (good) and rejected (bad) responses to the same prompt.
    Applies loss masking to only compute preference loss on assistant responses.
    
    Args:
        file_path: Path to JSONL with 'chosen' and 'rejected' fields
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
    """
    
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        
        # Load preference data
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a preference pair sample.
        
        Returns:
            Dictionary with tokenized chosen/rejected sequences and their loss masks
        """
        item = self.data[index]
        chosen = item['chosen']      # List of {role, content} dicts
        rejected = item['rejected']  # List of {role, content} dicts
        
        # Apply chat template to format conversations
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize and pad sequences
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length' #type: ignore 
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length' #type: ignore 
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids) #type: ignore 

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids) #type: ignore 
        
        # Convert to tensors
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long) #type: ignore 
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long) #type: ignore 
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long) #type: ignore 
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long) #type: ignore 
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        """Generate loss mask for preference data (same logic as SFT)."""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    Dataset for Reinforcement Learning from AI Feedback (RLAIF).
    
    Provides prompts and expected answers for online RL training methods
    like PPO, GRPO, and SPO. Unlike other datasets, this returns raw text
    rather than tokenized inputs.
    
    Args:
        jsonl_path: Path to JSONL with 'conversations' field
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length (not used directly here)
    """
    
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load conversation data from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Build ChatML-formatted prompt and extract expected answer.
        
        Args:
            conversations: List of conversation turns
            
        Returns:
            Tuple of (formatted_prompt, expected_answer)
        """
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        
        # Format prompt without the final assistant response
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # Add assistant marker at end
        )
        return cast(str, prompt), answer

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Get prompt and answer pair for RL training."""
        sample = self.samples[index]
        
        # Build conversation prompt and extract expected answer
        prompt, answer = self._create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    # Module test placeholder
    pass