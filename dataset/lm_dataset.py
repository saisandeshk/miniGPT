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
    Supports multiple source schemas:
      - {'conversations': [ {role, content}, ... ]}
      - Alpaca: {'instruction', 'input'?, 'output'}
      - Prompt-completion: {'prompt', 'completion'}
    """
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = self.load_data(jsonl_path)
        self.samples = raw  # keep raw, transform lazily
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def load_data(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                except json.JSONDecodeError:
                    continue
        return samples

    def _normalize_conversations(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        # Direct chat schema
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            conv = sample['conversations']
            if all(isinstance(m, dict) and 'role' in m and 'content' in m for m in conv):
                return conv

        # Alpaca schema
        if 'instruction' in sample and 'output' in sample:
            user_text = sample['instruction']
            if sample.get('input'):
                inp = sample['input']
                if inp and isinstance(inp, str) and inp.strip():
                    user_text = f"{user_text}\n{inp}"
            return [
                {'role': 'user', 'content': user_text},
                {'role': 'assistant', 'content': sample['output']}
            ]

        # Prompt-completion schema
        if 'prompt' in sample and 'completion' in sample:
            return [
                {'role': 'user', 'content': sample['prompt']},
                {'role': 'assistant', 'content': sample['completion']}
            ]

        # Simple text-only schema (your current dataset: keys ['text','source'])
        if 'text' in sample and isinstance(sample['text'], str):
            # Heuristic: first line (until first newline) = user prompt, remainder = assistant answer.
            txt = sample['text'].strip()
            if '\n' not in txt:
                # Single line: treat as user-only; create empty assistant (no loss)
                return [
                    {'role': 'user', 'content': txt},
                    {'role': 'assistant', 'content': ''}
                ]
            lines = txt.split('\n')
            user_part = lines[0].strip()
            assistant_part = '\n'.join(lines[1:]).strip()
            # Guard: if assistant_part accidentally empty, still return structure
            return [
                {'role': 'user', 'content': user_part},
                {'role': 'assistant', 'content': assistant_part}
            ]

        raise KeyError("Unsupported sample schema (keys: %s)" % list(sample.keys()))

    def _create_chat_prompt(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Build prompt. Use tokenizer.chat_template if available, else fallback.
        Fallback format keeps '<bos>assistant' contiguous so loss mask logic works.
        """
        # Fallback if no chat template defined
        if not getattr(self.tokenizer, "chat_template", None):
            parts = []
            for msg in conversations:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    # Optional: include user marker (not used by loss mask)
                    parts.append(f"{self.tokenizer.bos_token}user\n{content}\n")
                elif role == "assistant":
                    # Critical: bos + assistant (no space) so bos_id matches
                    parts.append(f"{self.tokenizer.bos_token}assistant\n{content}{self.tokenizer.eos_token}\n")
                else:
                    # Treat other roles as system/user prefix
                    parts.append(f"{self.tokenizer.bos_token}{role}\n{content}\n")
            return "".join(parts).strip()

        # Original path (chat template available)
        messages = conversations.copy()
        tools = messages[0].get("functions") if (messages and messages[0].get("role") == "system" and messages[0].get("functions")) else None
        return cast(str, self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        ))

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        conversations = self._normalize_conversations(sample)
        prompt = self._create_chat_prompt(conversations)

        tokenized = self.tokenizer(prompt)
        input_ids = tokenized.input_ids[:self.max_length]
        if len(input_ids) < self.max_length:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask_list = self._generate_loss_mask(input_ids)

        # Shift for causal LM
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask_list[1:], dtype=torch.long)

        return X, Y, loss_mask

# Modified DPO dataset class

class DPODataset(Dataset):
    """
    DPO dataset supporting multiple schemas:
      - {'prompt': str, 'chosen': str, 'rejected': str}
      - {'messages': [...], 'chosen': str, 'rejected': str}  (messages end with user)
      - {'chosen': [ {role, content}, ... ], 'rejected': [...]}
      - {'responses': {'chosen': str, 'rejected': str}, 'prompt': str?}
      - {'pairs': [ {'prompt': ..., 'chosen': ..., 'rejected': ...}, ... ] }
    Produces tensors:
      {'x_chosen','y_chosen','mask_chosen','x_rejected','y_rejected','mask_rejected'}
    Loss mask only covers assistant tokens, matching SFT behavior.
    """
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        raw = self._load_data(jsonl_path)

        # Precompute IDs for loss mask detection
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

        # Flatten to normalized pairs of conversations
        self.items: List[Dict[str, Any]] = []
        for sample in raw:
            for conv_pair in self._normalize_to_pairs(sample):
                self.items.append(conv_pair)

        if len(self.items) == 0:
            raise ValueError(f"No valid DPO pairs parsed from: {jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
        return data

    # ---------- schema normalization ----------
    def _first_key(self, d: Dict[str, Any], candidates: List[str]) -> Any:
        for k in candidates:
            if k in d and d[k] is not None:
                return d[k]
        return None

    def _ensure_messages(self, maybe_str_or_msgs: Any, prompt_text: str | None) -> List[Dict[str, str]]:
        # If already messages with role/content
        if isinstance(maybe_str_or_msgs, list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in maybe_str_or_msgs):
            return maybe_str_or_msgs
        # If it's a string, attach to optional prompt
        text = maybe_str_or_msgs if isinstance(maybe_str_or_msgs, str) else ""
        msgs: List[Dict[str, str]] = []
        if prompt_text is not None:
            msgs.append({'role': 'user', 'content': str(prompt_text)})
        msgs.append({'role': 'assistant', 'content': str(text)})
        return msgs

    def _normalize_to_pairs(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []

        # 1) Explicit pairs list
        if isinstance(sample.get('pairs'), list):
            for p in sample['pairs']:
                prompt_text = self._first_key(p, ['prompt', 'question', 'instruction', 'input', 'context', 'query'])
                chosen_val = self._first_key(p, ['chosen', 'chosen_text', 'chosen_response', 'response_chosen', 'preferred', 'better', 'pos'])
                rejected_val = self._first_key(p, ['rejected', 'rejected_text', 'rejected_response', 'response_rejected', 'worse', 'neg'])
                if chosen_val is None or rejected_val is None:
                    continue
                conv_chosen = self._ensure_messages(chosen_val, prompt_text)
                conv_rejected = self._ensure_messages(rejected_val, prompt_text)
                pairs.append({'chosen': conv_chosen, 'rejected': conv_rejected})
            if pairs:
                return pairs

        # 2) responses dict
        if isinstance(sample.get('responses'), dict):
            prompt_text = self._first_key(sample, ['prompt', 'question', 'instruction', 'input', 'context', 'query'])
            chosen_val = self._first_key(sample['responses'], ['chosen', 'preferred', 'better', 'pos'])
            rejected_val = self._first_key(sample['responses'], ['rejected', 'worse', 'neg'])
            if chosen_val is not None and rejected_val is not None:
                conv_chosen = self._ensure_messages(chosen_val, prompt_text)
                conv_rejected = self._ensure_messages(rejected_val, prompt_text)
                pairs.append({'chosen': conv_chosen, 'rejected': conv_rejected})
                return pairs

        # 3) chosen/rejected at top level (strings or message lists)
        chosen_val = self._first_key(sample, ['chosen', 'chosen_text', 'chosen_response', 'response_chosen', 'preferred', 'better', 'pos', 'messages_chosen'])
        rejected_val = self._first_key(sample, ['rejected', 'rejected_text', 'rejected_response', 'response_rejected', 'worse', 'neg', 'messages_rejected'])
        if chosen_val is not None and rejected_val is not None:
            # If base has a conversation context to prepend
            base_messages = None
            for ctx_key in ['messages', 'conversation', 'conversations']:
                if isinstance(sample.get(ctx_key), list):
                    maybe_msgs = sample[ctx_key]
                    if all(isinstance(m, dict) and 'role' in m and 'content' in m for m in maybe_msgs):
                        base_messages = maybe_msgs
                        break

            if base_messages is not None:
                # Append assistant completion to a copy of base_messages for each branch
                def with_completion(base, completion):
                    msgs = list(base)  # shallow copy
                    msgs.append({'role': 'assistant', 'content': completion if isinstance(completion, str) else ''})
                    return msgs

                conv_chosen = chosen_val if isinstance(chosen_val, list) else with_completion(base_messages, chosen_val)
                conv_rejected = rejected_val if isinstance(rejected_val, list) else with_completion(base_messages, rejected_val)
            else:
                # Use text prompt if available
                prompt_text = self._first_key(sample, ['prompt', 'question', 'instruction', 'input', 'context', 'query'])
                conv_chosen = self._ensure_messages(chosen_val, prompt_text)
                conv_rejected = self._ensure_messages(rejected_val, prompt_text)

            pairs.append({'chosen': conv_chosen, 'rejected': conv_rejected})
            return pairs

        # 4) Anthropic-like: {'contexts': [...], 'chosen': str, 'rejected': str}
        if isinstance(sample.get('contexts'), list) and all(isinstance(m, dict) and 'role' in m and 'content' in m for m in sample['contexts']):
            chosen_val = self._first_key(sample, ['chosen', 'preferred', 'better', 'pos'])
            rejected_val = self._first_key(sample, ['rejected', 'worse', 'neg'])
            if chosen_val is not None and rejected_val is not None:
                def append_to_context(ctx, text):
                    msgs = list(ctx)
                    msgs.append({'role': 'assistant', 'content': text})
                    return msgs
                pairs.append({
                    'chosen': append_to_context(sample['contexts'], chosen_val),
                    'rejected': append_to_context(sample['contexts'], rejected_val),
                })
                return pairs

        # If we reach here, no known schema
        # Print keys once for easier debugging
        if not hasattr(self, "_warned_dpo_schema"):
            self._warned_dpo_schema = True
            print(f"[DPODataset] Unsupported sample schema; keys={list(sample.keys())}")
        return pairs

    # ---------- prompt formatting and loss mask ----------
    def _create_chat_prompt(self, conversations: List[Dict[str, Any]]) -> str:
        """
        Use tokenizer.chat_template if available; otherwise a safe fallback that
        marks assistant spans with '<bos>assistant' ... '<eos>' to drive the loss mask.
        """
        if not getattr(self.tokenizer, "chat_template", None):
            parts = []
            for msg in conversations:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    parts.append(f"{self.tokenizer.bos_token}user\n{content}\n")
                elif role == "assistant":
                    parts.append(f"{self.tokenizer.bos_token}assistant\n{content}{self.tokenizer.eos_token}\n")
                else:
                    parts.append(f"{self.tokenizer.bos_token}{role}\n{content}\n")
            return "".join(parts).strip()

        messages = conversations.copy()
        tools = messages[0].get("functions") if (messages and messages[0].get("role") == "system" and messages[0].get("functions")) else None
        return cast(str, self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        ))

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        # Copy of SFT assistant-span mask
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

    def _encode_conversation(self, conv: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self._create_chat_prompt(conv)
        # Avoid adding extra special tokens; we already include BOS/EOS in the template or fallback
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids[:self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        mask_list = self._generate_loss_mask(ids)

        X = torch.tensor(ids[:-1], dtype=torch.long)
        Y = torch.tensor(ids[1:], dtype=torch.long)
        M = torch.tensor(mask_list[1:], dtype=torch.long)
        return X, Y, M

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        conv_chosen: List[Dict[str, Any]] = item['chosen']
        conv_rejected: List[Dict[str, Any]] = item['rejected']

        x_c, y_c, m_c = self._encode_conversation(conv_chosen)
        x_r, y_r, m_r = self._encode_conversation(conv_rejected)

        return {
            'x_chosen': x_c,
            'y_chosen': y_c,
            'mask_chosen': m_c,
            'x_rejected': x_r,
            'y_rejected': y_r,
            'mask_rejected': m_r,
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

    Returns:
      {'prompt': str}
    The prompt is formatted to end with an assistant cue so the model can generate a response.
    Supports multiple schemas:
      - {'conversations': [ {role, content}, ... ]}
      - {'messages': [ {role, content}, ... ]}
      - {'text': str}  (first line used as user prompt)
      - {'text': [ {role, content}, ... ]}  (treated like conversations)
      - {'prompt': str}
    """
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
        return data

    # ---------- schema normalization ----------
    def _ensure_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        # 1) Preferred: explicit message lists
        for key in ['conversations', 'messages', 'text']:
            if isinstance(sample.get(key), list):
                msgs = sample[key]
                if all(isinstance(m, dict) and 'role' in m and 'content' in m for m in msgs):
                    return msgs

        # 2) Single prompt string
        if isinstance(sample.get('prompt'), str) and sample['prompt'].strip():
            return [{'role': 'user', 'content': sample['prompt'].strip()}]

        # 3) Text string (SFT-like; first line as prompt)
        if isinstance(sample.get('text'), str):
            txt = sample['text'].strip()
            if txt:
                first_line = txt.split('\n', 1)[0].strip()
                return [{'role': 'user', 'content': first_line}]

        # Unsupported schema
        raise KeyError(f"Unsupported RLAIF sample schema; keys={list(sample.keys())}")

    def _trim_to_generation_prompt(self, msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure conversation ends at a user turn and append an assistant cue for generation.
        If the last message is assistant, drop it to force generation of a new assistant turn.
        """
        trimmed: List[Dict[str, str]] = []

        # Copy while ensuring valid keys
        for m in msgs:
            role = m.get('role', 'user')
            content = m.get('content', '')
            if not isinstance(content, str):
                content = str(content)
            trimmed.append({'role': role, 'content': content})

        # If ends with assistant content, remove final assistant to allow new assistant generation
        if trimmed and trimmed[-1]['role'] == 'assistant':
            trimmed = trimmed[:-1]

        # If empty or still ends with assistant (e.g., all assistant removed), ensure a user exists
        if not trimmed:
            trimmed = [{'role': 'user', 'content': ''}]
        elif trimmed[-1]['role'] != 'user':
            # If last isn't user, append a user stub to cue assistant next
            trimmed.append({'role': 'user', 'content': ''})

        return trimmed

    def _create_prompt_string(self, messages: List[Dict[str, Any]]) -> str:
        """
        Build a chat-formatted prompt that ends with an assistant cue.
        Uses tokenizer.chat_template if available, else a safe fallback that aligns with SFT.
        """
        msgs_for_prompt = self._trim_to_generation_prompt(messages)

        # If a chat template is available, use it and request a generation prompt
        if getattr(self.tokenizer, "chat_template", None):
            tools = msgs_for_prompt[0].get("functions") if (msgs_for_prompt and msgs_for_prompt[0].get("role") == "system" and msgs_for_prompt[0].get("functions")) else None
            return cast(str, self.tokenizer.apply_chat_template(
                msgs_for_prompt,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools
            ))

        # Fallback formatting (compatible with SFTDataset fallback)
        # We emit full history up to user, then an assistant header to cue generation.
        parts: List[str] = []
        bos = self.tokenizer.bos_token or ""
        eos = self.tokenizer.eos_token or ""

        for msg in msgs_for_prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"{bos}system\n{content}\n")
            elif role == "user":
                parts.append(f"{bos}user\n{content}\n")
            elif role == "assistant":
                # Keep historical assistant with EOS
                parts.append(f"{bos}assistant\n{content}{eos}\n")
            else:
                parts.append(f"{bos}{role}\n{content}\n")

        # Append assistant cue for generation (no EOS after header)
        parts.append(f"{bos}assistant\n")
        return "".join(parts).strip()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        messages = self._ensure_messages(sample)
        prompt = self._create_prompt_string(messages)
        # No tokenization here; PPO loop tokenizes with max_seq_len and left padding.
        return {"prompt": prompt}

if __name__ == "__main__":
    # Module test placeholder
    pass