# Implementation Guide: Getting Started

**Version:** 1.0  
**Date:** 2025-11-12  
**Audience:** Developers implementing the modular pipeline

---

## 🎯 Quick Start: Phase 1 Implementation

This guide provides step-by-step instructions for implementing **Phase 1: Dataset Mixture Infrastructure**.

---

## 📋 Prerequisites

### Required Skills
- Python 3.8+
- PyTorch basics
- YAML configuration
- Git version control

### System Requirements
- Python 3.8+
- CUDA 11.7+ (for GPU training)
- 50GB+ disk space (for datasets)

---

## 🚀 Step 1: Create Directory Structure

```bash
cd miniGPT

# Create config directories
mkdir -p config/data/pretrain/phase1
mkdir -p config/data/pretrain/phase2
mkdir -p config/data/midtrain/phase1
mkdir -p config/data/midtrain/phase2
mkdir -p config/data/posttrain/sft
mkdir -p config/data/posttrain/dpo
mkdir -p config/data/posttrain/ppo
mkdir -p config/data/posttrain/rlaif

mkdir -p config/model/minimind
mkdir -p config/model/llama
mkdir -p config/model/qwen
mkdir -p config/model/deepseek

mkdir -p config/experiments/templates
mkdir -p config/experiments/active

# Create new module directories
mkdir -p evaluation/benchmarks
mkdir -p evaluation/metrics
mkdir -p tests
mkdir -p notebooks

echo "✅ Directory structure created!"
```

---

## 🔧 Step 2: Implement Dataset Mixer Core

### 2.1 Create Mixture Config Schema

**File:** `config/data/pretrain/phase1/toy_mixture.yaml`

```yaml
metadata:
  phase: "pretrain_phase1"
  description: "Toy mixture for testing (100MB total)"
  total_tokens: 25_000_000  # 25M tokens
  max_seq_length: 512
  version: "1.0"

datasets:
  # Dataset 1: Sample from HuggingFace
  - name: "tinystories"
    source: "roneneldan/TinyStories"
    mix_ratio: 0.6  # 60% of tokens
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
    max_samples: 50000
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 10000

  # Dataset 2: Local JSONL file
  - name: "custom_data"
    source: "dataset/minimind_dataset/pretrain_sample.jsonl"
    mix_ratio: 0.4  # 40% of tokens
    format: "jsonl"
    text_field: "text"
    max_samples: 30000
    
    filters:
      - type: "length"
        min_length: 50

validation:
  ratio: 0.05  # 5% for validation
  seed: 42
  stratified: true  # Maintain mix ratios in validation set
```

### 2.2 Implement Mixer Core Logic

**File:** `dataset/mixer.py`

```python
"""
Dataset mixing engine for combining multiple datasets and saving to JSONL.

The mixer preprocesses and combines datasets according to mixture ratios,
then saves to a JSONL file for use by the unified dataset class.
"""

import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from datasets import load_dataset, Dataset as HFDataset


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in the mixture."""
    name: str
    source: str
    mix_ratio: float
    format: str  # 'huggingface', 'jsonl', 'parquet'
    text_field: str
    splits: List[str]
    max_samples: Optional[int] = None
    filters: List[Dict[str, Any]] = None


@dataclass
class MixtureConfig:
    """Configuration for the complete dataset mixture."""
    phase: str
    description: str
    total_tokens: int
    max_seq_length: int
    datasets: List[DatasetConfig]
    validation_ratio: float
    validation_seed: int


class DatasetMixer:
    """
    Main class for mixing multiple datasets and generating JSONL output.
    
    Usage:
        mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/mixture1.yaml")
        
        # Generate mixed JSONL file
        mixer.prepare_dataset(
            output_file="dataset/pretrain_phase1_mixture1.jsonl",
            split="train"
        )
        
        # Then use with existing dataset class:
        # dataset = PretrainDataset("dataset/pretrain_phase1_mixture1.jsonl", tokenizer)
    """
    
    def __init__(self, config: MixtureConfig):
        self.config = config
        self.datasets = []
        
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DatasetMixer':
        """Load mixer configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse metadata
        metadata = config_dict['metadata']
        
        # Parse dataset configs
        dataset_configs = []
        for ds_dict in config_dict['datasets']:
            dataset_configs.append(DatasetConfig(
                name=ds_dict['name'],
                source=ds_dict['source'],
                mix_ratio=ds_dict['mix_ratio'],
                format=ds_dict['format'],
                text_field=ds_dict['text_field'],
                splits=ds_dict.get('splits', ['train']),
                max_samples=ds_dict.get('max_samples'),
                filters=ds_dict.get('filters', [])
            ))
        
        # Parse validation config
        val_config = config_dict.get('validation', {})
        
        # Create MixtureConfig
        mixture_config = MixtureConfig(
            phase=metadata['phase'],
            description=metadata['description'],
            total_tokens=metadata['total_tokens'],
            max_seq_length=metadata['max_seq_length'],
            datasets=dataset_configs,
            validation_ratio=val_config.get('ratio', 0.05),
            validation_seed=val_config.get('seed', 42)
        )
        
        return cls(mixture_config)
    
    def prepare_dataset(
        self, 
        output_file: str,
        split: str = "train",
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Load datasets, mix them, and save to JSONL file.
        
        Args:
            output_file: Path to output JSONL file (e.g., "pretrain_phase1_mixture1.jsonl")
            split: 'train' or 'validation'
            cache_dir: Directory for caching downloaded datasets
            
        Returns:
            Path to generated JSONL file
        """
        from dataset.loader import load_single_dataset
        from dataset.filters import apply_filters
        
        print(f"Preparing {split} dataset...")
        print(f"Output: {output_file}")
        
        # Load and filter all datasets
        loaded_datasets = []
        for ds_config in self.config.datasets:
            print(f"\nLoading dataset: {ds_config.name}")
            
            # Load raw dataset
            raw_dataset = load_single_dataset(
                source=ds_config.source,
                format=ds_config.format,
                splits=ds_config.splits,
                cache_dir=cache_dir
            )
            
            # Apply filters
            if ds_config.filters:
                print(f"  Applying filters...")
                raw_dataset = apply_filters(raw_dataset, ds_config.filters)
            
            # Limit samples if specified
            if ds_config.max_samples:
                raw_dataset = raw_dataset.select(range(min(len(raw_dataset), ds_config.max_samples)))
            
            print(f"  Dataset size: {len(raw_dataset)} samples")
            
            loaded_datasets.append({
                'name': ds_config.name,
                'dataset': raw_dataset,
                'ratio': ds_config.mix_ratio,
                'text_field': ds_config.text_field
            })
        
        # Split into train/val
        if split == "validation":
            loaded_datasets = self._create_validation_split(loaded_datasets)
        else:
            loaded_datasets = self._create_train_split(loaded_datasets)
        
        # Mix and save to JSONL
        self._mix_and_save(loaded_datasets, output_file)
        
        return output_file
    
    def _create_train_split(self, datasets: List[Dict]) -> List[Dict]:
        """Create training split from loaded datasets."""
        train_datasets = []
        for ds_info in datasets:
            dataset = ds_info['dataset']
            val_size = int(len(dataset) * self.config.validation_ratio)
            train_size = len(dataset) - val_size
            
            train_ds = dataset.select(range(train_size))
            train_datasets.append({
                **ds_info,
                'dataset': train_ds
            })
        return train_datasets
    
    def _create_validation_split(self, datasets: List[Dict]) -> List[Dict]:
        """Create validation split from loaded datasets."""
        val_datasets = []
        for ds_info in datasets:
            dataset = ds_info['dataset']
            val_size = int(len(dataset) * self.config.validation_ratio)
            train_size = len(dataset) - val_size
            
            val_ds = dataset.select(range(train_size, len(dataset)))
            val_datasets.append({
                **ds_info,
                'dataset': val_ds
            })
        return val_datasets
    
    def _mix_and_save(self, datasets: List[Dict], output_file: str):
        """
        Mix datasets according to ratios and save to JSONL.
        
        Creates samples list with proper proportions, shuffles, and saves.
        """
        print("\nMixing datasets...")
        
        # Calculate total samples and samples per dataset
        total_samples = sum(len(ds['dataset']) for ds in datasets)
        samples_per_dataset = [
            int(total_samples * ds['ratio']) 
            for ds in datasets
        ]
        
        # Collect samples from each dataset
        all_samples = []
        
        for i, ds_info in enumerate(datasets):
            dataset = ds_info['dataset']
            text_field = ds_info['text_field']
            num_samples = samples_per_dataset[i]
            
            print(f"  {ds_info['name']}: {num_samples} samples ({ds_info['ratio']*100:.1f}%)")
            
            # Sample with wraparound if needed
            indices = []
            while len(indices) < num_samples:
                remaining = num_samples - len(indices)
                indices.extend(range(min(remaining, len(dataset))))
            
            # Extract text samples
            for idx in tqdm(indices, desc=f"  Processing {ds_info['name']}", leave=False):
                sample = dataset[idx % len(dataset)]
                text = sample[text_field]
                
                all_samples.append({
                    'text': text,
                    'source': ds_info['name']  # Track source for analysis
                })
        
        # Shuffle samples
        print(f"\nShuffling {len(all_samples)} samples...")
        random.seed(self.config.validation_seed)
        random.shuffle(all_samples)
        
        # Save to JSONL
        print(f"Saving to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(all_samples, desc="Writing"):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ Dataset saved: {output_file}")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def validate_mixture(self) -> Dict[str, float]:
        """
        Validate that mixture ratios sum to 1.0.
        
        Returns:
            Dictionary with validation results
        """
        total_ratio = sum(ds.mix_ratio for ds in self.config.datasets)
        
        return {
            'total_ratio': total_ratio,
            'is_valid': abs(total_ratio - 1.0) < 0.001,
            'individual_ratios': {ds.name: ds.mix_ratio for ds in self.config.datasets}
        }
```

### 2.3 Implement Dataset Loader

**File:** `dataset/loader.py`

```python
"""
Dataset loading utilities for various sources and formats.
"""

from typing import List, Optional, Union
from pathlib import Path

from datasets import load_dataset, Dataset


def load_single_dataset(
    source: str,
    format: str,
    splits: List[str] = ['train'],
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    Load a single dataset from various sources.
    
    Args:
        source: Dataset identifier (HF dataset name or local path)
        format: Format type ('huggingface', 'jsonl', 'parquet', 'arrow')
        splits: Which splits to load and concatenate
        cache_dir: Cache directory for downloads
        
    Returns:
        Loaded HuggingFace Dataset
    """
    if format == 'huggingface':
        # Load from HuggingFace Hub
        dataset = load_dataset(
            source,
            split='+'.join(splits),  # Concatenate splits
            cache_dir=cache_dir
        )
    
    elif format == 'jsonl':
        # Load from local JSONL file
        dataset = load_dataset(
            'json',
            data_files=source,
            split='train',
            cache_dir=cache_dir
        )
    
    elif format == 'parquet':
        # Load from Parquet file(s)
        dataset = load_dataset(
            'parquet',
            data_files=source,
            split='train',
            cache_dir=cache_dir
        )
    
    elif format == 'arrow':
        # Load from Arrow format
        dataset = Dataset.from_file(source)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return dataset
```

### 2.4 Implement Filters

**File:** `dataset/filters.py`

```python
"""
Dataset filtering and preprocessing utilities.
"""

from typing import List, Dict, Any
from datasets import Dataset


def apply_filters(dataset: Dataset, filters: List[Dict[str, Any]]) -> Dataset:
    """
    Apply a sequence of filters to a dataset.
    
    Args:
        dataset: Input dataset
        filters: List of filter configurations
        
    Returns:
        Filtered dataset
    """
    for filter_config in filters:
        filter_type = filter_config['type']
        
        if filter_type == 'length':
            dataset = filter_by_length(
                dataset,
                min_length=filter_config.get('min_length', 0),
                max_length=filter_config.get('max_length', float('inf'))
            )
        
        elif filter_type == 'language':
            dataset = filter_by_language(
                dataset,
                languages=filter_config['languages']
            )
        
        elif filter_type == 'quality':
            dataset = filter_by_quality(
                dataset,
                min_quality=filter_config.get('min_quality', 0.0)
            )
        
        # Add more filter types as needed
    
    return dataset


def filter_by_length(
    dataset: Dataset,
    min_length: int = 0,
    max_length: int = float('inf'),
    text_field: str = 'text'
) -> Dataset:
    """Filter dataset by text length (character count)."""
    def length_filter(example):
        text_len = len(example.get(text_field, ''))
        return min_length <= text_len <= max_length
    
    return dataset.filter(length_filter)


def filter_by_language(
    dataset: Dataset,
    languages: List[str],
    text_field: str = 'text'
) -> Dataset:
    """
    Filter dataset by language (requires langdetect package).
    
    Note: This is a placeholder. Implement actual language detection.
    """
    # TODO: Implement language detection
    # from langdetect import detect
    
    def language_filter(example):
        # Simplified: assume all text is in target language
        return True
    
    return dataset.filter(language_filter)


def filter_by_quality(
    dataset: Dataset,
    min_quality: float = 0.0,
    text_field: str = 'text'
) -> Dataset:
    """
    Filter dataset by quality score.
    
    Note: This is a placeholder. Implement actual quality scoring.
    """
    # TODO: Implement quality scoring (e.g., perplexity-based)
    
    def quality_filter(example):
        # Simplified: accept all for now
        return True
    
    return dataset.filter(quality_filter)
```

---

## 🧪 Step 3: Testing the Mixer

### 3.1 Create Test Script

**File:** `tests/test_mixer.py`

```python
"""
Unit tests for dataset mixer functionality.
"""

import pytest
from pathlib import Path
from transformers import AutoTokenizer

from dataset.mixer import DatasetMixer


def test_load_yaml_config():
    """Test loading mixer config from YAML."""
    yaml_path = "config/data/pretrain/phase1/toy_mixture.yaml"
    
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    assert mixer.config.phase == "pretrain_phase1"
    assert mixer.config.total_tokens == 25_000_000
    assert len(mixer.config.datasets) == 2


def test_validate_mixture():
    """Test mixture ratio validation."""
    yaml_path = "config/data/pretrain/phase1/toy_mixture.yaml"
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    validation = mixer.validate_mixture()
    
    assert validation['is_valid'] == True
    assert abs(validation['total_ratio'] - 1.0) < 0.001


def test_build_dataset():
    """Test building the mixed dataset."""
    yaml_path = "config/data/pretrain/phase1/toy_mixture.yaml"
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    tokenizer = AutoTokenizer.from_pretrained("model/")
    
    train_dataset = mixer.build_dataset(tokenizer, split="train")
    
    assert len(train_dataset) > 0
    
    # Test getting a sample
    sample = train_dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample


def test_train_val_split():
    """Test train/validation split."""
    yaml_path = "config/data/pretrain/phase1/toy_mixture.yaml"
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    tokenizer = AutoTokenizer.from_pretrained("model/")
    
    train_dataset = mixer.build_dataset(tokenizer, split="train")
    val_dataset = mixer.build_dataset(tokenizer, split="validation")
    
    # Validation should be smaller
    assert len(val_dataset) < len(train_dataset)
    
    # Approximate ratio check (allowing some tolerance)
    expected_ratio = mixer.config.validation_ratio
    actual_ratio = len(val_dataset) / (len(train_dataset) + len(val_dataset))
    assert abs(actual_ratio - expected_ratio) < 0.02  # Within 2%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 3.2 Run Tests

```bash
# Install pytest if needed
pip install pytest

# Run tests
cd miniGPT
python -m pytest tests/test_mixer.py -v
```

---

## 🔄 Step 4: Create Unified Dataset Class

### 4.1 Unified Dataset Class for All Phases

**File:** `dataset/lm_dataset.py` (update existing file)

Add a new unified dataset class that works for all training phases:

```python
class UnifiedDataset(Dataset):
    """
    Unified dataset class for all training phases.
    
    Works with preprocessed JSONL files generated by DatasetMixer.
    Handles: pretrain, midtrain, SFT, DPO, PPO, RLAIF
    
    Usage:
        # For pretraining
        dataset = UnifiedDataset(
            "dataset/pretrain_phase1_mixture1.jsonl",
            tokenizer,
            max_length=512,
            phase="pretrain"
        )
        
        # For SFT
        dataset = UnifiedDataset(
            "dataset/sft_general.jsonl",
            tokenizer,
            max_length=2048,
            phase="sft"
        )
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        phase: str = "pretrain"  # pretrain, sft, dpo, ppo, rlaif
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.phase = phase
        self.samples = self.load_data(data_path)
    
    def load_data(self, path: str) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """Get training sample based on phase."""
        sample = self.samples[index]
        
        if self.phase == "pretrain" or self.phase == "midtrain":
            return self._process_pretrain(sample)
        elif self.phase == "sft":
            return self._process_sft(sample)
        elif self.phase == "dpo":
            return self._process_dpo(sample)
        elif self.phase in ["ppo", "rlaif"]:
            return self._process_rl(sample)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
    
    def _process_pretrain(self, sample):
        """Process pretraining sample (simple next-token prediction)."""
        text = sample['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)
        
        # Create input-target pairs for causal LM
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        return X, Y, loss_mask
    
    def _process_sft(self, sample):
        """Process SFT sample (conversation format)."""
        # Existing SFTDataset logic
        conversations = sample.get('conversations', [])
        
        # Build full conversation text with special tokens
        full_text = ""
        loss_mask_ranges = []
        
        for turn in conversations:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                full_text += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                start_pos = len(full_text)
                full_text += f"<|assistant|>\n{content}\n"
                end_pos = len(full_text)
                loss_mask_ranges.append((start_pos, end_pos))
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze()
        
        # Create loss mask (only compute loss on assistant responses)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        for start, end in loss_mask_ranges:
            # Map character positions to token positions (approximate)
            loss_mask[start:end] = 1
        
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = loss_mask[1:]
        
        return X, Y, loss_mask
    
    def _process_dpo(self, sample):
        """Process DPO sample (chosen/rejected pairs)."""
        # Similar to existing DPODataset
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        # Tokenize all three
        prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
        chosen_enc = self.tokenizer(chosen, add_special_tokens=False)
        rejected_enc = self.tokenizer(rejected, add_special_tokens=False)
        
        # Combine and pad
        chosen_ids = prompt_enc['input_ids'] + chosen_enc['input_ids']
        rejected_ids = prompt_enc['input_ids'] + rejected_enc['input_ids']
        
        # Pad to max_length
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]
        
        # Create masks
        prompt_len = len(prompt_enc['input_ids'])
        chosen_mask = torch.zeros(len(chosen_ids), dtype=torch.long)
        chosen_mask[prompt_len:] = 1
        
        rejected_mask = torch.zeros(len(rejected_ids), dtype=torch.long)
        rejected_mask[prompt_len:] = 1
        
        return {
            'chosen_input_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_mask': chosen_mask,
            'rejected_mask': rejected_mask
        }
    
    def _process_rl(self, sample):
        """Process RL sample (for PPO/RLAIF)."""
        # Similar to existing RLAIFDataset
        prompt = sample['prompt']
        
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze()
        }
```

## 🚀 Step 5: Integration with Training

### 5.1 Update train_pretrain.py

**File:** `trainer/train_pretrain.py`

```python
# Add new command-line arguments
parser.add_argument('--data_config', type=str, default=None,
                    help='Path to dataset mixture YAML config')
parser.add_argument('--prepared_data', type=str, default=None,
                    help='Path to prepared JSONL file (skips preparation)')

# In main function, replace dataset loading:
if args.data_config:
    # Prepare dataset from mixture config
    from dataset.mixer import DatasetMixer
    from dataset.lm_dataset import UnifiedDataset
    
    print(f"Loading dataset mixture from: {args.data_config}")
    mixer = DatasetMixer.from_yaml(args.data_config)
    
    # Validate mixture
    validation = mixer.validate_mixture()
    print(f"Mixture validation: {validation}")
    assert validation['is_valid'], "Invalid mixture ratios!"
    
    # Generate JSONL files
    train_file = mixer.prepare_dataset(
        output_file=f"dataset/{mixer.config.phase}_train.jsonl",
        split="train"
    )
    
    val_file = mixer.prepare_dataset(
        output_file=f"dataset/{mixer.config.phase}_val.jsonl",
        split="validation"
    )
    
    # Load with unified dataset class
    train_dataset = UnifiedDataset(train_file, tokenizer, max_length=args.max_seq_len, phase="pretrain")
    val_dataset = UnifiedDataset(val_file, tokenizer, max_length=args.max_seq_len, phase="pretrain")

elif args.prepared_data:
    # Use pre-prepared JSONL file
    from dataset.lm_dataset import UnifiedDataset
    
    train_dataset = UnifiedDataset(args.prepared_data, tokenizer, max_length=args.max_seq_len, phase="pretrain")
    # Note: You'd need separate val file or split logic

else:
    # Use original dataset loading (backward compatibility)
    train_dataset = PretrainDataset(args.train_file, tokenizer, max_length=args.max_seq_len)
```

### 4.2 Test End-to-End

```bash
# Test with toy mixture
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/toy_mixture.yaml \
    --out_dir ./out \
    --epochs 1 \
    --batch_size 4 \
    --accumulation_steps 1 \
    --learning_rate 1e-4 \
    --device cuda:0
```

---

## 📊 Step 5: Create Example Mixtures

Create these example files to have ready-to-use mixtures:

1. **Toy Mixture** (100MB - for testing)
   - File: `config/data/pretrain/phase1/toy_mixture.yaml`
   - Already created above

2. **Small Mixture** (10GB - for small experiments)
   - File: `config/data/pretrain/phase1/small_mixture.yaml`

3. **Medium Mixture** (100GB - for validation)
   - File: `config/data/pretrain/phase1/medium_mixture.yaml`

4. **Full Mixture** (2TB - for actual 7B training)
   - File: `config/data/pretrain/phase1/full_mixture.yaml`

---

## ✅ Verification Checklist

After completing Phase 1 implementation, verify:

- [ ] All directories created
- [ ] `dataset/mixer.py` implemented and working
- [ ] `dataset/loader.py` supports HF and local files
- [ ] `dataset/filters.py` has basic filters
- [ ] Example YAML configs created
- [ ] Unit tests pass
- [ ] Integration with `train_pretrain.py` works
- [ ] Can train a small model with toy mixture
- [ ] Documentation updated

---

## 🐛 Common Issues & Solutions

### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'dataset.mixer'
```
**Solution:** Make sure `__init__.py` files exist in all directories:
```bash
touch dataset/__init__.py
touch evaluation/__init__.py
```

### Issue 2: YAML Parsing Errors
```
yaml.scanner.ScannerError: mapping values are not allowed here
```
**Solution:** Check YAML indentation (use spaces, not tabs)

### Issue 3: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `batch_size` or increase `gradient_accumulation_steps`

### Issue 4: Dataset Download Fails
```
ConnectionError: Couldn't reach HuggingFace servers
```
**Solution:** Check internet connection, try with `cache_dir` parameter

---

## 📚 Next Steps

After completing Phase 1:

1. **Phase 2:** Implement model registry and add Llama architecture
2. **Phase 3:** Create unified training CLI (`scripts/train.py`)
3. **Phase 4:** Add evaluation framework and benchmarks
4. **Phase 5:** Scale to 7B model training

---

## 🆘 Getting Help

- **Issues:** Check `docs/TODO.md` for known limitations
- **Questions:** Review `docs/ARCHITECTURE_PLAN.md` for design decisions
- **Debugging:** Enable verbose logging with `--verbose` flag

---

**Happy Coding! 🚀**
