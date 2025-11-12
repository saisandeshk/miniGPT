# Phase 1: Updated Approach - Preprocessing Pipeline

**Date:** 2025-11-12  
**Status:** Updated based on user feedback

---

## 🎯 Key Change: Preprocessing → JSONL → Training

### Original Approach
```
Mixture YAML → Mixer → On-the-fly sampling → PyTorch Dataset → Training
```

### **New Approach (Better!)**
```
Mixture YAML → Mixer → Preprocessed JSONL → UnifiedDataset → Training
                ↓
         pretrain_phase1_mixture1.jsonl
```

---

## ✅ Why This is Better

1. **Reproducibility**: Same JSONL = exact same training every time
2. **Inspectable**: Can examine mixed data before training
3. **Efficient**: Preprocess once, train multiple times
4. **Debuggable**: Easy to verify mixture ratios worked correctly
5. **Cacheable**: Keep JSONL files for repeated experiments
6. **Simpler**: Dataset class is just file I/O + tokenization

---

## 🔄 New Workflow

### Step 1: Prepare Dataset (One-time)

```python
from dataset.mixer import DatasetMixer

# Load mixture config
mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/mixture1.yaml")

# Validate mixture ratios
validation = mixer.validate_mixture()
print(validation)  # {'is_valid': True, 'total_ratio': 1.0, ...}

# Generate train JSONL
train_file = mixer.prepare_dataset(
    output_file="dataset/pretrain_phase1_mixture1_train.jsonl",
    split="train"
)

# Generate validation JSONL
val_file = mixer.prepare_dataset(
    output_file="dataset/pretrain_phase1_mixture1_val.jsonl",
    split="validation"
)

print(f"✅ Generated: {train_file} and {val_file}")
```

**Output JSONL format:**
```json
{"text": "This is sample text from dataset A...", "source": "tinystories"}
{"text": "Another sample from dataset B...", "source": "custom_data"}
{"text": "More training data...", "source": "tinystories"}
```

### Step 2: Train with UnifiedDataset

```python
from dataset.lm_dataset import UnifiedDataset

# Load prepared JSONL
train_dataset = UnifiedDataset(
    data_path="dataset/pretrain_phase1_mixture1_train.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    phase="pretrain"  # or "sft", "dpo", "ppo", "rlaif"
)

val_dataset = UnifiedDataset(
    data_path="dataset/pretrain_phase1_mixture1_val.jsonl",
    tokenizer=tokenizer,
    max_length=512,
    phase="pretrain"
)

# Use in DataLoader as normal
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for X, Y, loss_mask in train_loader:
    # X: input tokens
    # Y: target tokens (shifted by 1)
    # loss_mask: which tokens to compute loss on
    ...
```

---

## 📁 File Structure

```
miniGPT/
├── config/
│   └── data/
│       └── pretrain/
│           └── phase1/
│               ├── mixture1.yaml          # Mixture definition
│               ├── mixture2.yaml
│               └── toy_mixture.yaml
│
├── dataset/
│   ├── mixer.py                          # NEW: Preprocessing engine
│   ├── loader.py                         # NEW: Load source datasets
│   ├── filters.py                        # NEW: Quality filters
│   ├── lm_dataset.py                     # UPDATED: Add UnifiedDataset
│   │
│   └── (generated files)
│       ├── pretrain_phase1_mixture1_train.jsonl    ← Generated
│       ├── pretrain_phase1_mixture1_val.jsonl      ← Generated
│       ├── pretrain_phase1_mixture2_train.jsonl
│       ├── sft_general_train.jsonl
│       └── dpo_helpfulness_train.jsonl
│
└── trainer/
    └── train_pretrain.py                 # UPDATED: Use UnifiedDataset
```

---

## 🔧 Implementation Details

### DatasetMixer Class

**Key Methods:**

```python
class DatasetMixer:
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DatasetMixer':
        """Load mixture config from YAML."""
    
    def prepare_dataset(
        self,
        output_file: str,
        split: str = "train",
        cache_dir: Optional[str] = None
    ) -> str:
        """
        Load datasets, mix them, save to JSONL.
        
        Returns:
            Path to generated JSONL file
        """
        # 1. Load each dataset (HF, local JSONL, etc.)
        # 2. Apply filters (length, quality, language)
        # 3. Calculate samples per dataset based on mix_ratio
        # 4. Collect and shuffle samples
        # 5. Save to JSONL file
    
    def validate_mixture(self) -> Dict[str, float]:
        """Validate mixture ratios sum to 1.0."""
```

### UnifiedDataset Class

**Single class for ALL training phases:**

```python
class UnifiedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        phase: str = "pretrain"  # pretrain|sft|dpo|ppo|rlaif
    ):
        self.samples = self.load_data(data_path)  # Load JSONL
        self.phase = phase
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        
        if self.phase == "pretrain":
            return self._process_pretrain(sample)
        elif self.phase == "sft":
            return self._process_sft(sample)
        elif self.phase == "dpo":
            return self._process_dpo(sample)
        elif self.phase in ["ppo", "rlaif"]:
            return self._process_rl(sample)
    
    def _process_pretrain(self, sample):
        """Tokenize and create input/target pairs."""
        # Input: {"text": "...", "source": "..."}
        # Output: X, Y, loss_mask
    
    def _process_sft(self, sample):
        """Handle conversation format with role-based masking."""
        # Input: {"conversations": [{"role": "user", "content": "..."}, ...]}
        # Output: X, Y, loss_mask (only compute loss on assistant turns)
    
    def _process_dpo(self, sample):
        """Handle chosen/rejected pairs."""
        # Input: {"prompt": "...", "chosen": "...", "rejected": "..."}
        # Output: dict with chosen_ids, rejected_ids, masks
    
    def _process_rl(self, sample):
        """Handle prompts for RL training."""
        # Input: {"prompt": "..."}
        # Output: dict with input_ids, attention_mask
```

---

## 🚀 Updated Training Commands

### Quick Test (with toy mixture)

```bash
# Prepare toy dataset
python scripts/prepare_dataset.py \
    --mixture_config config/data/pretrain/phase1/toy_mixture.yaml \
    --output_dir dataset/

# Train
python trainer/train_pretrain.py \
    --prepared_data dataset/pretrain_phase1_toy_train.jsonl \
    --out_dir ./out_test \
    --epochs 1 \
    --batch_size 4

# Or do both in one command (auto-prepare if JSONL doesn't exist)
python trainer/train_pretrain.py \
    --data_config config/data/pretrain/phase1/toy_mixture.yaml \
    --out_dir ./out_test \
    --epochs 1
```

### Inspect Generated Data

```bash
# Look at generated JSONL
head -10 dataset/pretrain_phase1_mixture1_train.jsonl

# Count lines (number of samples)
wc -l dataset/pretrain_phase1_mixture1_train.jsonl

# Check file size
du -h dataset/pretrain_phase1_mixture1_train.jsonl

# Verify source distribution
cat dataset/pretrain_phase1_mixture1_train.jsonl | \
    jq -r '.source' | sort | uniq -c
```

### Reuse Prepared Data

```bash
# Prepare once
python scripts/prepare_dataset.py \
    --mixture_config config/data/pretrain/phase1/mixture1.yaml \
    --output_dir dataset/

# Train multiple times with same data
python trainer/train_pretrain.py \
    --prepared_data dataset/pretrain_phase1_mixture1_train.jsonl \
    --out_dir ./exp001 \
    --learning_rate 1e-4

python trainer/train_pretrain.py \
    --prepared_data dataset/pretrain_phase1_mixture1_train.jsonl \
    --out_dir ./exp002 \
    --learning_rate 5e-4

# Compare results!
```

---

## 📊 Benefits Summary

| Aspect | Old (On-the-fly) | New (Preprocessing) |
|--------|------------------|---------------------|
| **Reproducibility** | ⚠️ Stochastic sampling | ✅ Exact same data |
| **Inspectability** | ❌ Can't see mixed data | ✅ Can examine JSONL |
| **Efficiency** | ❌ Re-mix every run | ✅ Preprocess once |
| **Debugging** | ❌ Hard to verify mixing | ✅ Easy to check ratios |
| **Disk usage** | ✅ No extra files | ⚠️ JSONL files (manageable) |
| **Complexity** | 🟡 Complex Dataset class | ✅ Simple file I/O |

---

## 🎯 Updated Phase 1 Tasks

### Week 1: Core Implementation
- [x] Design updated approach (this document)
- [ ] Implement `dataset/mixer.py` with `prepare_dataset()`
- [ ] Implement `dataset/loader.py` (no change)
- [ ] Implement `dataset/filters.py` (no change)
- [ ] Add `UnifiedDataset` to `dataset/lm_dataset.py`

### Week 2: Integration & Validation
- [ ] Create `scripts/prepare_dataset.py` CLI tool
- [ ] Update `trainer/train_pretrain.py` to use prepared data
- [ ] Test with toy mixture
- [ ] Verify JSONL files are generated correctly
- [ ] Validate mixture ratios in output

---

## 🔍 Example Mixture → JSONL Flow

**Input:** `config/data/pretrain/phase1/toy_mixture.yaml`
```yaml
metadata:
  phase: "pretrain_phase1"
  
datasets:
  - name: "tinystories"
    mix_ratio: 0.6  # 60%
    max_samples: 50000
  
  - name: "custom_data"
    mix_ratio: 0.4  # 40%
    max_samples: 30000
```

**Process:**
1. Load 50k samples from TinyStories
2. Load 30k samples from custom_data
3. Total: 80k samples
4. Calculate: TinyStories gets 48k slots (60%), custom gets 32k slots (40%)
5. Sample with wraparound if needed
6. Shuffle all 80k samples
7. Save to JSONL

**Output:** `dataset/pretrain_phase1_toy_train.jsonl`
```json
{"text": "Once upon a time...", "source": "tinystories"}
{"text": "Custom sample...", "source": "custom_data"}
{"text": "Another story...", "source": "tinystories"}
...
(80,000 lines total)
```

**Verification:**
```bash
# Should show ~60% tinystories, ~40% custom_data
cat dataset/pretrain_phase1_toy_train.jsonl | \
    jq -r '.source' | sort | uniq -c

# Output:
#  48000 tinystories
#  32000 custom_data
```

---

## 🆘 FAQ

**Q: Do I need to regenerate JSONL every time?**  
A: No! Generate once, train multiple times with different hyperparameters.

**Q: What if I change the mixture config?**  
A: Re-run `prepare_dataset()` to generate new JSONL files with different names.

**Q: How much disk space do JSONL files take?**  
A: Roughly same as original compressed data. 10GB dataset → ~10GB JSONL.

**Q: Can I use the same UnifiedDataset for SFT?**  
A: Yes! Just pass `phase="sft"` and it handles conversation format automatically.

**Q: What about the old PretrainDataset class?**  
A: Keep it for backward compatibility, but new code uses UnifiedDataset.

---

## ✅ Summary

**Key Changes:**
1. ✅ Mixer now **preprocesses** and **saves to JSONL**
2. ✅ **UnifiedDataset** loads JSONL and works for ALL phases
3. ✅ Training scripts use prepared JSONL files
4. ✅ Can inspect mixed data before training
5. ✅ Fully reproducible experiments

**Next Steps:**
1. Update `dataset/mixer.py` with new `prepare_dataset()` method
2. Add `UnifiedDataset` to `dataset/lm_dataset.py`
3. Create `scripts/prepare_dataset.py` CLI tool
4. Test end-to-end flow

---

**Status:** Ready for implementation ✅  
**Priority:** HIGH - Foundation for all future work
