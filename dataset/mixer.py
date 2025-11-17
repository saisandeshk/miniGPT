"""
Dataset mixing engine for combining multiple datasets and saving to JSONL.

The mixer preprocesses and combines datasets according to mixture ratios,
then saves to a JSONL file for use by the unified dataset classes.
"""

import json
import yaml
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from datasets import Dataset as HFDataset


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
    filters: Optional[List[Dict[str, Any]]] = None


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
        mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/default.yaml")
        
        # Generate mixed JSONL file
        train_file = mixer.prepare_dataset(
            output_file="dataset/pretrain_phase1_default_train.jsonl",
            split="train"
        )
        
        # Then use with dataset class:
        # dataset = PretrainDataset("dataset/pretrain_phase1_default_train.jsonl", tokenizer)
    """
    
    def __init__(self, config: MixtureConfig):
        self.config = config
        
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
            output_file: Path to output JSONL file (e.g., "pretrain_phase1_default_train.jsonl")
            split: 'train' or 'validation'
            cache_dir: Directory for caching downloaded datasets
            
        Returns:
            Path to generated JSONL file
        """
        from dataset.loader import load_single_dataset
        from dataset.filters import apply_filters
        
        print(f"\n{'='*70}")
        print(f"Preparing {split} dataset...")
        print(f"Output: {output_file}")
        print(f"{'='*70}\n")
        
        # Load and filter all datasets
        loaded_datasets = []
        for ds_config in self.config.datasets:
            print(f"📦 Loading dataset: {ds_config.name}")
            print(f"   Source: {ds_config.source}")
            
            # Load raw dataset
            raw_dataset = load_single_dataset(
                source=ds_config.source,
                format=ds_config.format,
                splits=ds_config.splits,
                cache_dir=cache_dir
            )
            
            print(f"   Initial size: {len(raw_dataset)} samples")
            
            # Apply format conversion if needed
            from dataset.loader import convert_dataset_format
            raw_dataset = convert_dataset_format(raw_dataset, ds_config.source)
            
            # Apply filters
            if ds_config.filters:
                print(f"   Applying {len(ds_config.filters)} filter(s)...")
                raw_dataset = apply_filters(raw_dataset, ds_config.filters, ds_config.text_field)
            
            # Limit samples if specified
            if ds_config.max_samples and len(raw_dataset) > ds_config.max_samples:
                print(f"   Limiting to {ds_config.max_samples} samples")
                raw_dataset = raw_dataset.select(range(ds_config.max_samples))
            
            print(f"   ✅ Final size: {len(raw_dataset)} samples\n")
            
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
            
            if train_size > 0:
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
            
            if val_size > 0:
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
        print(f"🔀 Mixing datasets...")
        
        # Calculate total samples and samples per dataset
        total_samples = sum(len(ds['dataset']) for ds in datasets)
        samples_per_dataset = [
            int(total_samples * ds['ratio']) 
            for ds in datasets
        ]
        
        print(f"\n📊 Mixture composition:")
        for i, ds_info in enumerate(datasets):
            num_samples = samples_per_dataset[i]
            percentage = ds_info['ratio'] * 100
            print(f"   {ds_info['name']}: {num_samples:,} samples ({percentage:.1f}%)")
        
        # Collect samples from each dataset
        all_samples = []
        
        for i, ds_info in enumerate(datasets):
            dataset = ds_info['dataset']
            text_field = ds_info['text_field']
            num_samples = samples_per_dataset[i]
            
            # Sample with wraparound if needed
            indices = []
            while len(indices) < num_samples:
                remaining = num_samples - len(indices)
                indices.extend(range(min(remaining, len(dataset))))
            
            # Extract text samples
            for idx in tqdm(indices, desc=f"   Processing {ds_info['name']}", leave=False):
                sample = dataset[idx % len(dataset)]
                
                # Check if this is a DPO dataset (has chosen/rejected fields)
                if 'chosen' in sample and 'rejected' in sample:
                    # DPO format: preserve chosen and rejected
                    all_samples.append({
                        'chosen': sample['chosen'],
                        'rejected': sample['rejected'],
                        'source': ds_info['name']
                    })
                else:
                    # Regular format: extract text field
                    text = sample[text_field]
                    all_samples.append({
                        'text': text,
                        'source': ds_info['name']  # Track source for analysis
                    })
        
        # Shuffle samples
        print(f"\n🔀 Shuffling {len(all_samples):,} samples...")
        random.seed(self.config.validation_seed)
        random.shuffle(all_samples)
        
        # Save to JSONL
        print(f"💾 Saving to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(all_samples, desc="   Writing", leave=False):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Print summary
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\n{'='*70}")
        print(f"✅ Dataset saved successfully!")
        print(f"   File: {output_file}")
        print(f"   Samples: {len(all_samples):,}")
        print(f"   Size: {file_size_mb:.2f} MB")
        print(f"{'='*70}\n")
    
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
