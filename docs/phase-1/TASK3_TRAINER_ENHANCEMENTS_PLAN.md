# Task 3: Enhance Training Scripts - Implementation Plan

**Date:** 2025-11-16  
**Status:** 📝 Planning  
**Prerequisites:** ✅ Task 2 Complete (All configs ready)

---

## 🎯 Objective

Enhance post-training scripts to use dataset configs (like `train_pretrain.py` and `train_midtrain.py`), adding WandB logging, evaluation loops, and comprehensive metrics.

---

## 📊 Training Scripts Analysis

### Currently Available Scripts (10 total)

| Script | Size | Dataset Class | Priority | Action |
|--------|------|---------------|----------|--------|
| `train_pretrain.py` | 25K | PretrainDataset | ✅ | DONE (Task 1) |
| `train_midtrain.py` | 28K | PretrainDataset | ✅ | DONE (Task 1) |
| `train_full_sft.py` | 14K | SFTDataset | HIGH | **ENHANCE** |
| `train_dpo.py` | 19K | DPODataset | HIGH | **ENHANCE** |
| `train_ppo.py` | 30K | RLAIFDataset | HIGH | **ENHANCE** |
| `train_grpo.py` | 26K | RLAIFDataset | MEDIUM | **ENHANCE** |
| `train_spo.py` | 31K | RLAIFDataset | MEDIUM | **ENHANCE** |
| `train_lora.py` | 15K | SFTDataset | LOW | Skip for now |
| `train_distillation.py` | 18K | SFTDataset | LOW | Skip for now |
| `train_distill_reason.py` | 15K | SFTDataset | LOW | Skip for now |

---

## 📋 Scripts to Enhance (Priority Order)

### **Priority 1: Essential Post-Training Methods** (MUST DO)

#### 1. `train_full_sft.py` ⭐⭐⭐
- **Why**: Most important post-training step
- **Uses**: `SFTDataset` (conversations format)
- **Config**: `config/data/posttrain/sft/*.yaml`
- **Time**: ~30 min

#### 2. `train_dpo.py` ⭐⭐⭐
- **Why**: Standard preference alignment method
- **Uses**: `DPODataset` (chosen/rejected pairs)
- **Config**: `config/data/posttrain/dpo/*.yaml`
- **Time**: ~30 min

#### 3. `train_ppo.py` ⭐⭐⭐
- **Why**: Online RL method, commonly used
- **Uses**: `RLAIFDataset` (prompts)
- **Config**: `config/data/posttrain/rlaif/ppo.yaml`
- **Time**: ~30 min

### **Priority 2: Advanced RLAIF Methods** (NICE TO HAVE)

#### 4. `train_grpo.py` ⭐⭐
- **Why**: Advanced group-based RL
- **Uses**: `RLAIFDataset`
- **Config**: `config/data/posttrain/rlaif/grpo.yaml`
- **Time**: ~20 min

#### 5. `train_spo.py` ⭐⭐
- **Why**: Cutting-edge self-play method
- **Uses**: `RLAIFDataset`
- **Config**: `config/data/posttrain/rlaif/spo.yaml`
- **Time**: ~20 min

### **Priority 3: Optional Methods** (SKIP FOR NOW)

- `train_lora.py` - LoRA fine-tuning (different paradigm)
- `train_distillation.py` - Model distillation (not core pipeline)
- `train_distill_reason.py` - Reasoning distillation (specialized)

**Decision**: Focus on 5 scripts (SFT, DPO, PPO, GRPO, SPO)

---

## 🔧 What to Add to Each Script

Based on `train_pretrain.py` and `train_midtrain.py` as templates:

### **1. Argument Parser Additions**

```python
# Data mixture support
parser.add_argument("--data_config", type=str, default=None,
                    help="Path to dataset mixture YAML config")
parser.add_argument("--use_prepared", action="store_true",
                    help="Use pre-prepared JSONL")

# Evaluation support
parser.add_argument("--eval_interval", type=int, default=500,
                    help="Evaluation interval (steps), 0 to disable")
parser.add_argument("--eval_batches", type=int, default=100,
                    help="Number of batches for evaluation")

# WandB defaults updated
parser.add_argument("--wandb_project", type=str, 
                    default="MiniMind-[SFT|DPO|PPO|etc]",
                    help="WandB project name")
```

### **2. Dataset Preparation Logic**

```python
# Support two modes:
# 1. Direct data path (original)
# 2. Dataset config with mixer (new)

if args.data_config:
    from dataset.mixer import DatasetMixer
    
    print(f"Using dataset mixture config: {args.data_config}")
    mixer = DatasetMixer.from_yaml(args.data_config)
    
    # Validate
    validation = mixer.validate_mixture()
    if not validation['is_valid']:
        raise ValueError("Invalid mixture ratios!")
    
    # Generate filenames
    config_name = Path(args.data_config).stem
    train_jsonl = f"dataset/{mixer.config.phase}_{config_name}_train.jsonl"
    val_jsonl = f"dataset/{mixer.config.phase}_{config_name}_val.jsonl"
    
    # Prepare if needed
    if not args.use_prepared or not os.path.exists(train_jsonl):
        if is_main_process():
            print("Preparing dataset...")
            mixer.prepare_dataset(train_jsonl, split="train")
            mixer.prepare_dataset(val_jsonl, split="validation")
        
        if dist.is_initialized():
            dist.barrier()
    
    args.data_path = train_jsonl  # Use prepared file
```

### **3. Evaluation Function**

```python
def evaluate(model, eval_loader, device, autocast_ctx, max_batches=None):
    """
    Evaluate model on validation set.
    
    Note: Adapt loss calculation for each training type:
    - SFT: cross-entropy loss
    - DPO: preference loss
    - PPO: policy loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Compute loss (method-specific)
            loss = compute_loss(model, batch_data, device)
            
            total_loss += loss.item()
            total_tokens += get_token_count(batch_data)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    model.train()
    
    return {'eval/loss': avg_loss, ...}
```

### **4. Enhanced Logging in train_epoch**

```python
# Add to logging section
if wandb:
    wandb.log({
        # Training metrics
        "train/loss": current_loss,
        "train/learning_rate": current_lr,
        "train/grad_norm": grad_norm,
        "train/epoch": epoch + (step / iters),
        "train/global_step": global_step,
        
        # Throughput metrics
        "train/tokens_per_second": tokens_per_sec,
        "train/samples_per_second": samples_per_sec,
        "train/mfu_percent": mfu,
        
        # Eval metrics (if available)
        **eval_metrics
    }, step=global_step)
```

### **5. Evaluation Loop Integration**

```python
# In train_epoch, add periodic evaluation
if eval_loader and args.eval_interval > 0:
    if step % args.eval_interval == 0 or step == iters - 1:
        if is_main_process():
            Logger("Running evaluation...")
        
        eval_metrics = evaluate(
            model, eval_loader, args.device, 
            autocast_ctx, max_batches=args.eval_batches
        )
        
        if is_main_process():
            Logger(f"Eval loss: {eval_metrics['eval/loss']:.6f}")
```

### **6. Updated Docstrings**

Update all docstrings to reflect:
- Dataset config support
- WandB logging
- Evaluation capabilities
- Usage examples with `--data_config`

---

## 🔄 Implementation Strategy

### For Each Script:

1. **Copy core enhancements** from `train_pretrain.py`:
   - Argument additions
   - Dataset preparation logic
   - Evaluation function
   - Enhanced logging
   - Evaluation loop

2. **Adapt method-specific parts**:
   - Loss calculation (SFT vs DPO vs PPO)
   - Evaluation metrics (method-specific)
   - WandB project name
   - Default hyperparameters

3. **Keep method-specific logic intact**:
   - DPO: preference loss, reference model
   - PPO: policy/value networks, reward model
   - GRPO: group-based rewards
   - SPO: self-play logic

4. **Test minimally**:
   - Config loads
   - Script runs without errors
   - Basic functionality preserved

---

## 📊 Detailed Enhancement Plan per Script

### **Script 1: train_full_sft.py**

**Current state**: Basic SFT with cross-entropy loss

**Enhancements needed**:
- ✅ Add `--data_config` and `--use_prepared`
- ✅ Add dataset mixer integration
- ✅ Add evaluation function
- ✅ Add WandB comprehensive logging
- ✅ Add eval loop to train_epoch
- ✅ Update default `--wandb_project` to "MiniMind-SFT"
- ✅ Add MFU calculation
- ✅ Add throughput metrics

**Unique considerations**:
- SFT uses `SFTDataset` which expects `conversations` field
- Loss is standard cross-entropy on assistant responses
- Evaluation is straightforward

---

### **Script 2: train_dpo.py**

**Current state**: DPO with preference loss, reference model

**Enhancements needed**:
- ✅ Add `--data_config` and `--use_prepared`
- ✅ Add dataset mixer integration
- ✅ Add evaluation function (DPO loss specific)
- ✅ Add WandB logging
- ✅ Add eval loop
- ✅ Update `--wandb_project` to "MiniMind-DPO"

**Unique considerations**:
- DPO uses both policy and reference model
- Loss is preference-based (log ratio)
- Evaluation needs both models
- Dataset is `DPODataset` (chosen/rejected pairs)

---

### **Script 3: train_ppo.py**

**Current state**: PPO with policy/value networks, reward model

**Enhancements needed**:
- ✅ Add `--data_config` and `--use_prepared`
- ✅ Add dataset mixer integration
- ✅ Add evaluation function (PPO metrics)
- ✅ Add WandB logging (policy loss, value loss, rewards)
- ✅ Add eval loop
- ✅ Update `--wandb_project` to "MiniMind-PPO"

**Unique considerations**:
- PPO is more complex (actor-critic)
- Evaluation includes multiple metrics (policy loss, value loss, KL)
- Dataset is `RLAIFDataset` (prompts only)
- Online generation during training

---

### **Script 4: train_grpo.py**

**Current state**: Group-based RL with reward ranking

**Enhancements needed**:
- ✅ Add `--data_config` and `--use_prepared`
- ✅ Add dataset mixer integration
- ✅ Add evaluation (GRPO specific)
- ✅ Add WandB logging
- ✅ Update `--wandb_project` to "MiniMind-GRPO"

**Unique considerations**:
- Group-based reward comparison
- Multiple samples per prompt
- Similar to PPO but different loss

---

### **Script 5: train_spo.py**

**Current state**: Self-play optimization

**Enhancements needed**:
- ✅ Add `--data_config` and `--use_prepared`
- ✅ Add dataset mixer integration
- ✅ Add evaluation (SPO specific)
- ✅ Add WandB logging
- ✅ Update `--wandb_project` to "MiniMind-SPO"

**Unique considerations**:
- Self-play against previous versions
- Potentially more complex evaluation
- Most cutting-edge method

---

## ⏱️ Time Estimates

| Script | Time | Complexity | Priority |
|--------|------|------------|----------|
| train_full_sft.py | 30 min | Low | HIGH |
| train_dpo.py | 30 min | Medium | HIGH |
| train_ppo.py | 30 min | Medium | HIGH |
| train_grpo.py | 20 min | Medium | MEDIUM |
| train_spo.py | 20 min | High | MEDIUM |

**Total: ~2-2.5 hours** for all 5 scripts

**Recommendation**: Start with Priority 1 (SFT, DPO, PPO) = ~1.5 hours

---

## ✅ Success Criteria

For each enhanced script:

1. **Functionality**:
   - [ ] Accepts `--data_config` argument
   - [ ] Loads dataset mixture correctly
   - [ ] Supports `--use_prepared` flag
   - [ ] Has evaluation function
   - [ ] Logs to WandB comprehensively

2. **Testing**:
   - [ ] Script imports without errors
   - [ ] Config loads and validates
   - [ ] Help text is updated
   - [ ] Basic syntax is correct

3. **Documentation**:
   - [ ] Docstrings updated
   - [ ] Usage examples added
   - [ ] Comments explain new features

4. **Integration**:
   - [ ] Compatible with Task 2 configs
   - [ ] Works with mixer pipeline
   - [ ] Consistent with pretrain/midtrain style

---

## 🚀 Implementation Order

### Session 1: Core Scripts (~1.5 hours)
1. **train_full_sft.py** (30 min) - Most important
2. **train_dpo.py** (30 min) - Standard alignment
3. **train_ppo.py** (30 min) - Online RL baseline

### Session 2: Advanced Scripts (~1 hour) [OPTIONAL]
4. **train_grpo.py** (20 min) - Advanced RL
5. **train_spo.py** (20 min) - Cutting-edge
6. **Documentation** (20 min) - Completion doc

---

## 📝 Testing Approach

### Per-Script Testing

```bash
# 1. Test help text
python trainer/train_full_sft.py --help | grep data_config

# 2. Test config loading
python -c "
import sys
sys.argv = ['script', '--data_config', 'config/data/posttrain/sft/general.yaml', '--help']
import trainer.train_full_sft
"

# 3. Test syntax
python -m py_compile trainer/train_full_sft.py
```

### Integration Testing (Quick)

```bash
# Test SFT with config (dry run)
python trainer/train_full_sft.py \
    --data_config config/data/posttrain/sft/general.yaml \
    --from_weight out/midtrain_512.pth \
    --epochs 1 --batch_size 4 \
    --device cpu \
    --help
```

---

## 🎯 Key Decisions

### Decision 1: Which Scripts to Enhance?
**Choice**: 5 scripts (SFT, DPO, PPO, GRPO, SPO)
**Rationale**: Covers all essential post-training methods, skip specialized (LoRA, distillation)

### Decision 2: How Much to Copy?
**Choice**: Copy argument parser, dataset prep, eval function, logging from `train_pretrain.py`
**Rationale**: Consistency across all scripts, proven approach

### Decision 3: How Much to Test?
**Choice**: Minimal testing (syntax, imports, help text)
**Rationale**: Pipeline focus, not production quality yet

### Decision 4: Priority Order?
**Choice**: SFT → DPO → PPO (Priority 1), then GRPO → SPO (Priority 2)
**Rationale**: Cover essential methods first, advanced methods are optional

---

## 📂 Expected Output

After Task 3:

```
trainer/
├── train_pretrain.py ✅ (Task 1)
├── train_midtrain.py ✅ (Task 1)
├── train_full_sft.py ✅ (Task 3) - Enhanced
├── train_dpo.py ✅ (Task 3) - Enhanced
├── train_ppo.py ✅ (Task 3) - Enhanced
├── train_grpo.py ✅ (Task 3) - Enhanced [Optional]
├── train_spo.py ✅ (Task 3) - Enhanced [Optional]
├── train_lora.py ⏸️ (Unchanged for now)
├── train_distillation.py ⏸️ (Unchanged for now)
└── train_distill_reason.py ⏸️ (Unchanged for now)
```

---

## 🤔 Review Questions

Before starting implementation:

1. **Script selection OK?**
   - Include: SFT, DPO, PPO, GRPO, SPO
   - Exclude: LoRA, Distillation scripts

2. **Priority order correct?**
   - Start with SFT/DPO/PPO?
   - GRPO/SPO optional?

3. **Enhancement scope appropriate?**
   - Add config support ✓
   - Add evaluation ✓
   - Add WandB logging ✓
   - Keep method-specific logic intact ✓

4. **Timeline realistic?**
   - ~30 min per Priority 1 script
   - ~20 min per Priority 2 script
   - Total ~2-2.5 hours

---

**Status:** 📝 **AWAITING APPROVAL TO START**

**Recommendation:** Start with Priority 1 (SFT, DPO, PPO), then decide on Priority 2 based on time/need.

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-16
