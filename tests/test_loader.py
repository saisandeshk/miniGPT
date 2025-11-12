"""
Unit tests for dataset loading functionality.
"""

import pytest
from dataset.loader import load_single_dataset, get_dataset_info


def test_load_huggingface_dataset():
    """Test loading from HuggingFace Hub."""
    # Use a small, well-known dataset
    dataset = load_single_dataset(
        source="roneneldan/TinyStories",
        format="huggingface",
        splits=["train"]
    )
    
    assert len(dataset) > 0
    assert 'text' in dataset.column_names


def test_get_dataset_info():
    """Test getting dataset information."""
    info = get_dataset_info("roneneldan/TinyStories", format="huggingface")
    
    assert 'source' in info
    assert 'format' in info
    assert info['source'] == "roneneldan/TinyStories"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
