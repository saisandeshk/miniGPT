"""
Dataset loading utilities for various sources and formats.

Currently focuses on HuggingFace datasets with basic support for local files.
"""

from typing import List, Optional
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
        
    Examples:
        >>> dataset = load_single_dataset("roneneldan/TinyStories", "huggingface")
        >>> print(len(dataset))
    """
    if format == 'huggingface':
        # Load from HuggingFace Hub
        try:
            # Concatenate multiple splits if needed
            if len(splits) == 1:
                dataset = load_dataset(
                    source,
                    split=splits[0],
                    cache_dir=cache_dir,
                    trust_remote_code=True  # Some datasets need this
                )
            else:
                # Load and concatenate multiple splits
                datasets = []
                for split in splits:
                    ds = load_dataset(
                        source,
                        split=split,
                        cache_dir=cache_dir,
                        trust_remote_code=True
                    )
                    datasets.append(ds)
                
                # Concatenate all splits
                from datasets import concatenate_datasets
                dataset = concatenate_datasets(datasets)
            
            return dataset
            
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace dataset '{source}': {str(e)}")
    
    elif format == 'jsonl':
        # Load from local JSONL file
        try:
            dataset = load_dataset(
                'json',
                data_files=source,
                split='train',
                cache_dir=cache_dir
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load JSONL file '{source}': {str(e)}")
    
    elif format == 'parquet':
        # Load from Parquet file(s)
        try:
            dataset = load_dataset(
                'parquet',
                data_files=source,
                split='train',
                cache_dir=cache_dir
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Parquet file '{source}': {str(e)}")
    
    elif format == 'arrow':
        # Load from Arrow format
        try:
            dataset = Dataset.from_file(source)
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load Arrow file '{source}': {str(e)}")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: huggingface, jsonl, parquet, arrow")


def get_dataset_info(source: str, format: str = 'huggingface') -> dict:
    """
    Get information about a dataset without loading it fully.
    
    Args:
        source: Dataset identifier
        format: Dataset format
        
    Returns:
        Dictionary with dataset metadata
    """
    if format == 'huggingface':
        try:
            from datasets import get_dataset_config_names, get_dataset_split_names
            
            info = {
                'source': source,
                'format': format,
                'configs': [],
                'splits': []
            }
            
            try:
                info['configs'] = get_dataset_config_names(source)
            except:
                info['configs'] = ['default']
            
            try:
                info['splits'] = get_dataset_split_names(source)
            except:
                info['splits'] = ['train']
            
            return info
        except Exception as e:
            return {
                'source': source,
                'format': format,
                'error': str(e)
            }
    
    return {'source': source, 'format': format, 'info': 'unavailable'}
