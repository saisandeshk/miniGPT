#!/usr/bin/env python3
"""
Pre-flight check script to verify Phase 1 implementation is ready.

This script checks:
1. All required files exist
2. Python syntax is valid
3. Imports work correctly
4. YAML config is valid
5. DatasetMixer can be instantiated

Usage:
    python scripts/preflight_check.py
"""

import os
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_file_exists(filepath, min_size=100):
    """Check if file exists and has content."""
    if not os.path.exists(filepath):
        return False, "File not found"
    
    size = os.path.getsize(filepath)
    if size < min_size:
        return False, f"File too small ({size} bytes)"
    
    return True, f"OK ({size:,} bytes)"


def check_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def check_imports():
    """Test all imports."""
    results = {}
    
    # Check loader
    try:
        from dataset.loader import load_single_dataset, get_dataset_info
        results['dataset.loader'] = (True, "OK")
    except Exception as e:
        results['dataset.loader'] = (False, str(e))
    
    # Check mixer
    try:
        from dataset.mixer import DatasetMixer
        results['dataset.mixer'] = (True, "OK")
    except Exception as e:
        results['dataset.mixer'] = (False, str(e))
    
    # Check filters
    try:
        from dataset.filters import apply_filters, calculate_quality_score
        results['dataset.filters'] = (True, "OK")
    except Exception as e:
        results['dataset.filters'] = (False, str(e))
    
    # Check lm_dataset
    try:
        from dataset.lm_dataset import PretrainDataset
        results['dataset.lm_dataset'] = (True, "OK")
    except Exception as e:
        results['dataset.lm_dataset'] = (False, str(e))
    
    return results


def check_yaml_config(filepath):
    """Validate YAML config."""
    try:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check structure
        if 'metadata' not in config:
            return False, "Missing 'metadata' section"
        
        if 'datasets' not in config:
            return False, "Missing 'datasets' section"
        
        # Check ratios
        total_ratio = sum(ds['mix_ratio'] for ds in config['datasets'])
        if abs(total_ratio - 1.0) > 0.001:
            return False, f"Ratios sum to {total_ratio}, not 1.0"
        
        return True, f"Valid ({len(config['datasets'])} datasets)"
    
    except Exception as e:
        return False, str(e)


def check_mixer():
    """Test DatasetMixer instantiation."""
    try:
        from dataset.mixer import DatasetMixer
        
        mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/default.yaml")
        validation = mixer.validate_mixture()
        
        if not validation['is_valid']:
            return False, "Mixture validation failed"
        
        return True, "Mixer ready"
    
    except Exception as e:
        return False, str(e)


def main():
    print("\n" + "=" * 70)
    print("PHASE 1 PRE-FLIGHT CHECK")
    print("=" * 70 + "\n")
    
    all_passed = True
    
    # Check files
    print("1. Checking files exist...")
    print("-" * 70)
    
    files = [
        "config/data/pretrain/phase1/default.yaml",
        "config/data/README.md",
        "dataset/loader.py",
        "dataset/mixer.py",
        "dataset/filters.py",
        "tests/test_mixer.py",
        "tests/test_filters.py",
        "tests/test_loader.py",
        "scripts/prepare_dataset.py",
        "scripts/test_mixer_pipeline.py",
    ]
    
    for filepath in files:
        ok, msg = check_file_exists(filepath)
        status = "✅" if ok else "❌"
        print(f"  {status} {filepath:45s} {msg}")
        if not ok:
            all_passed = False
    
    # Check Python syntax
    print("\n2. Checking Python syntax...")
    print("-" * 70)
    
    python_files = [f for f in files if f.endswith('.py')]
    
    for filepath in python_files:
        ok, msg = check_python_syntax(filepath)
        status = "✅" if ok else "❌"
        print(f"  {status} {filepath:45s} {msg}")
        if not ok:
            all_passed = False
    
    # Check imports
    print("\n3. Checking imports...")
    print("-" * 70)
    
    import_results = check_imports()
    
    for module, (ok, msg) in import_results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {module:45s} {msg}")
        if not ok:
            all_passed = False
    
    # Check YAML config
    print("\n4. Checking YAML config...")
    print("-" * 70)
    
    ok, msg = check_yaml_config("config/data/pretrain/phase1/default.yaml")
    status = "✅" if ok else "❌"
    print(f"  {status} {'config/data/pretrain/phase1/default.yaml':45s} {msg}")
    if not ok:
        all_passed = False
    
    # Check mixer
    print("\n5. Checking DatasetMixer...")
    print("-" * 70)
    
    ok, msg = check_mixer()
    status = "✅" if ok else "❌"
    print(f"  {status} {'DatasetMixer.from_yaml()':45s} {msg}")
    if not ok:
        all_passed = False
    
    # Final result
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED! Ready to test.")
        print("=" * 70 + "\n")
        print("Next steps:")
        print("  1. Run: python scripts/test_mixer_pipeline.py")
        print("  2. Run: python -m pytest tests/ -v")
        print("  3. Try: python scripts/prepare_dataset.py --config config/data/pretrain/phase1/default.yaml")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED! Please fix errors above.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
