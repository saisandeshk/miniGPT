#!/usr/bin/env python3
"""
Quick test script to verify the dataset mixer pipeline works end-to-end.

Usage:
    python scripts/test_mixer_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.mixer import DatasetMixer
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer


def test_mixer_pipeline():
    """Test the complete mixer pipeline."""
    
    print("\n" + "="*70)
    print("Testing Dataset Mixer Pipeline")
    print("="*70 + "\n")
    
    # Step 1: Load mixer config
    print("Step 1: Loading mixer configuration...")
    config_path = "config/data/pretrain/phase1/default.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    mixer = DatasetMixer.from_yaml(config_path)
    print(f"✅ Loaded config: {mixer.config.phase}")
    print(f"   Description: {mixer.config.description}")
    print(f"   Datasets: {len(mixer.config.datasets)}")
    
    # Step 2: Validate mixture
    print("\nStep 2: Validating mixture ratios...")
    validation = mixer.validate_mixture()
    print(f"   Total ratio: {validation['total_ratio']:.3f}")
    print(f"   Is valid: {validation['is_valid']}")
    
    if not validation['is_valid']:
        print("❌ Invalid mixture ratios!")
        return False
    
    print("✅ Mixture validated")
    
    # Step 3: Prepare dataset
    print("\nStep 3: Preparing dataset (this may take a few minutes)...")
    output_file = "dataset/test_pretrain_phase1_default_train.jsonl"
    
    train_file = mixer.prepare_dataset(
        output_file=output_file,
        split="train"
    )
    
    if not os.path.exists(train_file):
        print(f"❌ Failed to create {train_file}")
        return False
    
    print(f"✅ Dataset prepared: {train_file}")
    
    # Step 4: Load with PretrainDataset
    print("\nStep 4: Loading with PretrainDataset...")
    
    # Load tokenizer
    tokenizer_path = "model/"
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("   Please make sure the model directory exists with tokenizer files")
        return False
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load dataset
    dataset = PretrainDataset(train_file, tokenizer, max_length=512)
    
    print(f"✅ Dataset loaded with {len(dataset)} samples")
    
    # Step 5: Test getting a sample
    print("\nStep 5: Testing sample retrieval...")
    if len(dataset) > 0:
        X, Y, loss_mask = dataset[0]
        print(f"✅ Sample retrieved successfully")
        print(f"   Input shape: {X.shape}")
        print(f"   Target shape: {Y.shape}")
        print(f"   Loss mask shape: {loss_mask.shape}")
        print(f"   Non-padding tokens: {loss_mask.sum().item()}")
    else:
        print("❌ Dataset is empty!")
        return False
    
    # Success!
    print("\n" + "="*70)
    print("✅ All tests passed! Pipeline is working correctly.")
    print("="*70 + "\n")
    
    # Cleanup
    print("Cleaning up test file...")
    if os.path.exists(train_file):
        os.remove(train_file)
        print(f"✅ Removed {train_file}")
    
    return True


if __name__ == "__main__":
    success = test_mixer_pipeline()
    sys.exit(0 if success else 1)
