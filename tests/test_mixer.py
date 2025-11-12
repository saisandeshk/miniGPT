"""
Unit tests for dataset mixer functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path

from dataset.mixer import DatasetMixer


def test_load_yaml_config():
    """Test loading mixer config from YAML."""
    yaml_path = "config/data/pretrain/phase1/default.yaml"
    
    if not os.path.exists(yaml_path):
        pytest.skip(f"Config file not found: {yaml_path}")
    
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    assert mixer.config.phase == "pretrain_phase1"
    assert mixer.config.total_tokens == 50_000_000
    assert len(mixer.config.datasets) >= 1


def test_validate_mixture():
    """Test mixture ratio validation."""
    yaml_path = "config/data/pretrain/phase1/default.yaml"
    
    if not os.path.exists(yaml_path):
        pytest.skip(f"Config file not found: {yaml_path}")
    
    mixer = DatasetMixer.from_yaml(yaml_path)
    validation = mixer.validate_mixture()
    
    assert validation['is_valid'] == True
    assert abs(validation['total_ratio'] - 1.0) < 0.001


def test_prepare_dataset():
    """Test dataset preparation (generates JSONL file)."""
    yaml_path = "config/data/pretrain/phase1/default.yaml"
    
    if not os.path.exists(yaml_path):
        pytest.skip(f"Config file not found: {yaml_path}")
    
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    # Use temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "test_train.jsonl")
        
        # This will download and process the dataset
        result_file = mixer.prepare_dataset(
            output_file=output_file,
            split="train"
        )
        
        assert os.path.exists(result_file)
        assert os.path.getsize(result_file) > 0
        
        # Check JSONL format
        import json
        with open(result_file, 'r') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            assert 'text' in data
            assert 'source' in data


def test_train_val_split():
    """Test train/validation split."""
    yaml_path = "config/data/pretrain/phase1/default.yaml"
    
    if not os.path.exists(yaml_path):
        pytest.skip(f"Config file not found: {yaml_path}")
    
    mixer = DatasetMixer.from_yaml(yaml_path)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = os.path.join(tmpdir, "train.jsonl")
        val_file = os.path.join(tmpdir, "val.jsonl")
        
        mixer.prepare_dataset(train_file, split="train")
        mixer.prepare_dataset(val_file, split="validation")
        
        assert os.path.exists(train_file)
        assert os.path.exists(val_file)
        
        # Count lines
        with open(train_file) as f:
            train_lines = sum(1 for _ in f)
        with open(val_file) as f:
            val_lines = sum(1 for _ in f)
        
        # Validation should be smaller
        assert val_lines < train_lines
        
        # Check approximate ratio
        total_lines = train_lines + val_lines
        val_ratio = val_lines / total_lines
        expected_ratio = mixer.config.validation_ratio
        
        # Allow some tolerance
        assert abs(val_ratio - expected_ratio) < 0.02


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
