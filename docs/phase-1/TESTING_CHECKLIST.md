# Phase 1 Testing Checklist

Use this checklist to verify the Phase 1 implementation works correctly.

---

## ✅ Pre-Testing Setup

- [ ] Changed to miniGPT directory: `cd /home/saisandeshk/llm/miniGPT`
- [ ] Python 3.7+ available: `python3 --version`
- [ ] Required packages installed: `pip list | grep -E "(torch|transformers|datasets|yaml)"`

---

## 🔍 Step 1: Pre-Flight Check (2 minutes)

**Command:**
```bash
python scripts/preflight_check.py
```

**Expected:** All checks pass with ✅

**Checklist:**
- [ ] All 10 files exist
- [ ] All Python files have valid syntax
- [ ] All imports work
- [ ] YAML config is valid
- [ ] DatasetMixer loads successfully

**If failed:** Check error messages and fix issues before proceeding.

---

## 🧪 Step 2: Unit Tests (5-10 minutes)

### Test Filters
```bash
python -m pytest tests/test_filters.py -v
```
- [ ] test_filter_by_length passes
- [ ] test_calculate_quality_score passes
- [ ] test_filter_by_quality passes
- [ ] test_apply_filters passes

### Test Loader
```bash
python -m pytest tests/test_loader.py -v
```
- [ ] test_load_huggingface_dataset passes
- [ ] test_get_dataset_info passes

**Note:** First run will download TinyStories (~2.5GB). Subsequent runs use cache.

### Test Mixer
```bash
python -m pytest tests/test_mixer.py -v
```
- [ ] test_load_yaml_config passes
- [ ] test_validate_mixture passes
- [ ] test_prepare_dataset passes
- [ ] test_train_val_split passes

**Expected time:** 5-10 minutes (depends on download speed)

---

## 🔄 Step 3: End-to-End Pipeline Test (10-15 minutes)

**Command:**
```bash
python scripts/test_mixer_pipeline.py
```

**What it does:**
1. Loads mixer config
2. Validates mixture
3. Prepares a test JSONL file
4. Loads it with PretrainDataset
5. Retrieves a sample
6. Cleans up

**Checklist:**
- [ ] Step 1: Config loads successfully
- [ ] Step 2: Mixture validated
- [ ] Step 3: Dataset prepared (JSONL created)
- [ ] Step 4: PretrainDataset loads JSONL
- [ ] Step 5: Sample retrieved with correct shapes
- [ ] All tests passed message shown

**If tokenizer error:** Make sure `model/` directory exists with tokenizer files.

---

## 📦 Step 4: Dataset Preparation (10-20 minutes)

**Command:**
```bash
python scripts/prepare_dataset.py \
    --config config/data/pretrain/phase1/default.yaml \
    --output_dir dataset/
```

**Expected output files:**
- `dataset/pretrain_phase1_default_train.jsonl` (~12-15 MB)
- `dataset/pretrain_phase1_default_val.jsonl` (~1-2 MB)

**Checklist:**
- [ ] TinyStories downloaded (or used from cache)
- [ ] Filters applied successfully
- [ ] Train dataset created (~95,000 samples)
- [ ] Validation dataset created (~5,000 samples)
- [ ] Both files exist and have content

**Verify:**
```bash
# Check files exist
ls -lh dataset/pretrain_phase1_default_*.jsonl

# Count samples
wc -l dataset/pretrain_phase1_default_train.jsonl
wc -l dataset/pretrain_phase1_default_val.jsonl

# Look at first sample
head -1 dataset/pretrain_phase1_default_train.jsonl
```

---

## 🚂 Step 5: Training Integration Test (time varies)

### Option A: Quick Smoke Test (5 minutes)
```bash
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --use_prepared \
    --epochs 1 \
    --batch_size 4 \
    --accumulation_steps 1 \
    --learning_rate 1e-4 \
    --device cpu \
    --max_seq_len 128
```

**Checklist:**
- [ ] Mixer config loaded
- [ ] Mixture validated
- [ ] Pre-prepared dataset used
- [ ] Training starts without errors
- [ ] At least one batch completes
- [ ] Loss is computed

### Option B: GPU Training Test (depends on hardware)
```bash
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/default.yaml \
    --use_prepared \
    --epochs 1 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --device cuda:0
```

**Checklist:**
- [ ] GPU detected and used
- [ ] Training completes at least 10 steps
- [ ] Loss decreases
- [ ] GPU utilization >50%
- [ ] No OOM errors

---

## 🔧 Step 6: Custom Mixture Test (15 minutes)

Create a custom mixture config:

```bash
cp config/data/pretrain/phase1/default.yaml \
   config/data/pretrain/phase1/test_mixture.yaml
```

Edit `test_mixture.yaml` if you want different settings.

**Prepare:**
```bash
python scripts/prepare_dataset.py \
    --config config/data/pretrain/phase1/test_mixture.yaml \
    --output_dir dataset/
```

**Checklist:**
- [ ] Custom config loads
- [ ] New JSONL files created
- [ ] Can train with custom mixture

---

## 📊 Step 7: Validation (5 minutes)

### Check JSONL Format
```bash
# Should show JSON with "text" and "source" fields
head -5 dataset/pretrain_phase1_default_train.jsonl | python -m json.tool
```

**Checklist:**
- [ ] Valid JSON format
- [ ] Has "text" field
- [ ] Has "source" field
- [ ] Text content looks reasonable

### Verify Mix Ratios
```bash
# Count samples per source (requires jq)
cat dataset/pretrain_phase1_default_train.jsonl | \
    jq -r '.source' | sort | uniq -c
```

**Expected:** 95,000 tinystories samples (100%)

**Checklist:**
- [ ] Ratios match config
- [ ] All sources present
- [ ] Counts are reasonable

### Check Dataset Loading
```python
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model/")
dataset = PretrainDataset(
    "dataset/pretrain_phase1_default_train.jsonl",
    tokenizer,
    max_length=512
)

print(f"Dataset size: {len(dataset)}")
X, Y, loss_mask = dataset[0]
print(f"Sample shapes: X={X.shape}, Y={Y.shape}, mask={loss_mask.shape}")
```

**Checklist:**
- [ ] Dataset loads without errors
- [ ] Length matches expected (~95,000)
- [ ] Samples have correct shapes
- [ ] Loss mask identifies non-padding tokens

---

## 🎯 Step 8: Backward Compatibility Test

Test that original mode still works:

```bash
python trainer/train_pretrain.py \
    --data_path dataset/pretrain_phase1_default_train.jsonl \
    --epochs 1 \
    --batch_size 4 \
    --device cpu
```

**Checklist:**
- [ ] Works without --data_config
- [ ] No mixer messages printed
- [ ] Training proceeds normally

---

## 📝 Final Verification

### All Green?
- [ ] Pre-flight check passed
- [ ] All unit tests passed
- [ ] End-to-end test passed
- [ ] Dataset preparation successful
- [ ] Training integration works
- [ ] Custom mixture works
- [ ] Validation checks out
- [ ] Backward compatibility confirmed

### Files Created
- [ ] `dataset/pretrain_phase1_default_train.jsonl`
- [ ] `dataset/pretrain_phase1_default_val.jsonl`
- [ ] Any custom mixture JSONL files

---

## 🐛 Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'dataset'"
**Solution:** Make sure you're in the miniGPT directory
```bash
cd /home/saisandeshk/llm/miniGPT
```

### Issue: "FileNotFoundError: config/data/pretrain/phase1/default.yaml"
**Solution:** Check you're running from the correct directory
```bash
pwd  # Should show /home/saisandeshk/llm/miniGPT
ls config/data/pretrain/phase1/default.yaml  # Should exist
```

### Issue: "Failed to load HuggingFace dataset"
**Solution:** Check internet connection or try loading manually:
```python
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")
```

### Issue: "Tokenizer not found"
**Solution:** Verify model directory exists:
```bash
ls model/tokenizer_config.json
```

### Issue: Out of memory during dataset prep
**Solution:** Reduce `max_samples` in YAML config temporarily

### Issue: Tests take too long
**Solution:** Tests download TinyStories first time (~2.5GB). Use cache after.

---

## ✅ Success Criteria

Phase 1 is successfully implemented if:

1. ✅ Pre-flight check passes completely
2. ✅ All unit tests pass
3. ✅ End-to-end pipeline test completes
4. ✅ Can prepare datasets from YAML config
5. ✅ Can train with prepared datasets
6. ✅ JSONL files have correct format
7. ✅ Mix ratios are accurate
8. ✅ Backward compatibility maintained

---

## 🎉 Completion

Once all checks pass, you're ready for:
- Creating custom dataset mixtures
- Experimenting with different data compositions
- Moving to Phase 2 (Model Architecture Modularity)

**Congratulations! Phase 1 is complete and verified! 🚀**

---

**Date Completed:** _____________  
**Tested By:** _____________  
**Status:** ⬜ Not Started | ⬜ In Progress | ⬜ Complete ✅
