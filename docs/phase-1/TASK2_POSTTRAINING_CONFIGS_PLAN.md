# Task 2: Post-Training Data Configs - Implementation Plan

**Date:** 2025-11-15  
**Status:** 📝 Planning Phase  
**Prerequisites:** ✅ Task 1 Complete (Mid-training working)

---

## 🎯 Objective

Create YAML configuration files for all post-training stages (SFT, DPO, PPO/RLAIF) using HuggingFace datasets, ensuring compatibility with existing dataset classes and mixer infrastructure.

---

## 📊 Current State Analysis

### ✅ What's Already Working

1. **Dataset Classes** (in `dataset/lm_dataset.py`):
   - ✅ `SFTDataset` - expects `conversations` field
   - ✅ `DPODataset` - expects `chosen` and `rejected` fields
   - ✅ `RLAIFDataset` - expects `conversations` field

2. **Dataset Mixer**:
   - ✅ Loads from YAML configs
   - ✅ Applies filters
   - ✅ Generates JSONL files
   - ✅ Handles multiple data sources

3. **Filters Available**:
   - `length` - character-based filtering
   - `quality` - heuristic quality score
   - `code_quality` - code content detection
   - `language` - (placeholder)

### 🔨 What Needs to be Created

Post-training configs for:
1. **SFT** (Supervised Fine-Tuning) - 3 configs
2. **DPO** (Direct Preference Optimization) - 2 configs
3. **RLAIF** (PPO/GRPO/SPO) - 3 configs

---

## 📋 Implementation Plan

### **Step 1: Research HuggingFace Datasets** ⏰ 15 min

Find appropriate datasets for each stage:

#### SFT Datasets (Conversational format)
- **General**: instruction-following, Q&A, chat
- **Code**: coding instructions, code explanation
- **Reasoning**: math, logic, chain-of-thought

**Good HF datasets for SFT:**
- `HuggingFaceH4/ultrachat_200k` - General chat (good quality)
- `Open-Orca/OpenOrca` - Diverse instructions (large)
- `WizardLM/WizardLM_evol_instruct_V2_196k` - Complex instructions
- `theblackcat102/evol-codealpaca-v1` - Code-focused
- `TIGER-Lab/MathInstruct` - Math reasoning
- `tatsu-lab/alpaca` - Simple instructions (classic)

#### DPO Datasets (Preference pairs)
- **Helpfulness**: chosen vs rejected responses
- **Safety**: safe vs unsafe responses

**Good HF datasets for DPO:**
- `Anthropic/hh-rlhf` - Helpfulness & harmlessness preferences
- `HuggingFaceH4/ultrafeedback_binarized` - Quality preferences
- `argilla/ultrafeedback-binarized-preferences` - Curated preferences
- `Intel/orca_dpo_pairs` - High-quality pairs

#### RLAIF Datasets (Prompt-based)
- **PPO/GRPO/SPO**: prompts for online learning

**Good HF datasets for RLAIF:**
- `HuggingFaceH4/ultrachat_200k` - Can extract prompts
- `OpenAssistant/oasst1` - Multi-turn conversations
- `lmsys/chatbot_arena_conversations` - Real user prompts

---

### **Step 2: Create SFT Configs** ⏰ 30 min

Create 3 SFT mixture configs emphasizing different skills.

#### 2.1 General SFT (`config/data/posttrain/sft/general.yaml`)

**Goal:** Broad instruction following across domains

```yaml
metadata:
  phase: "sft_general"
  description: "General instruction fine-tuning with diverse tasks"
  total_tokens: 10_000_000  # 10M tokens
  max_seq_length: 512
  version: "1.0"

datasets:
  # 50% - General instruction following
  - name: "ultrachat_general"
    source: "HuggingFaceH4/ultrachat_200k"
    mix_ratio: 0.5
    format: "huggingface"
    text_field: "messages"  # Will need to reformat to 'conversations'
    splits: ["train_sft"]
    max_samples: 50000
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 2000
      
      - type: "quality"
        min_score: 0.3
  
  # 30% - Instruction diversity (OpenOrca)
  - name: "openorca_instructions"
    source: "Open-Orca/OpenOrca"
    mix_ratio: 0.3
    format: "huggingface"
    text_field: "conversations"  # Will format from system/question/response
    splits: ["train"]
    max_samples: 30000
    
    filters:
      - type: "length"
        min_length: 30
        max_length: 2000
      
      - type: "quality"
        min_score: 0.2
  
  # 20% - Simple Alpaca-style instructions
  - name: "alpaca_instructions"
    source: "tatsu-lab/alpaca"
    mix_ratio: 0.2
    format: "huggingface"
    text_field: "text"  # Will format from instruction/output
    splits: ["train"]
    max_samples: 20000
    
    filters:
      - type: "length"
        min_length: 20
        max_length: 1500

validation:
  ratio: 0.05
  seed: 42
  stratified: true
```

#### 2.2 Code SFT (`config/data/posttrain/sft/code.yaml`)

**Goal:** Programming and code generation skills

```yaml
metadata:
  phase: "sft_code"
  description: "Code-focused instruction fine-tuning"
  total_tokens: 8_000_000
  max_seq_length: 1024  # Longer for code
  version: "1.0"

datasets:
  # 70% - Code generation instructions
  - name: "evol_codealpaca"
    source: "theblackcat102/evol-codealpaca-v1"
    mix_ratio: 0.7
    format: "huggingface"
    text_field: "output"
    splits: ["train"]
    max_samples: 35000
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 3000
      
      - type: "code_quality"
        min_code_ratio: 0.15
  
  # 30% - General instructions with some code
  - name: "ultrachat_code_subset"
    source: "HuggingFaceH4/ultrachat_200k"
    mix_ratio: 0.3
    format: "huggingface"
    text_field: "messages"
    splits: ["train_sft"]
    max_samples: 15000
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 2000

validation:
  ratio: 0.05
  seed: 42
```

#### 2.3 Reasoning SFT (`config/data/posttrain/sft/reasoning.yaml`)

**Goal:** Mathematical and logical reasoning

```yaml
metadata:
  phase: "sft_reasoning"
  description: "Math and reasoning instruction fine-tuning"
  total_tokens: 6_000_000
  max_seq_length: 1024
  version: "1.0"

datasets:
  # 60% - Math instructions
  - name: "math_instruct"
    source: "TIGER-Lab/MathInstruct"
    mix_ratio: 0.6
    format: "huggingface"
    text_field: "instruction"
    splits: ["train"]
    max_samples: 36000
    
    filters:
      - type: "length"
        min_length: 30
        max_length: 2000
      
      - type: "quality"
        min_score: 0.3
  
  # 40% - General reasoning
  - name: "ultrachat_reasoning"
    source: "HuggingFaceH4/ultrachat_200k"
    mix_ratio: 0.4
    format: "huggingface"
    text_field: "messages"
    splits: ["train_sft"]
    max_samples: 24000
    
    filters:
      - type: "length"
        min_length: 40
        max_length: 2000

validation:
  ratio: 0.05
  seed: 42
```

---

### **Step 3: Create DPO Configs** ⏰ 20 min

Create 2 DPO configs for preference alignment.

#### 3.1 Helpfulness DPO (`config/data/posttrain/dpo/helpfulness.yaml`)

**Goal:** Align model to produce helpful, high-quality responses

```yaml
metadata:
  phase: "dpo_helpfulness"
  description: "Preference optimization for helpfulness and quality"
  total_tokens: 5_000_000
  max_seq_length: 1024
  version: "1.0"

datasets:
  # 60% - High-quality preference data
  - name: "ultrafeedback_binarized"
    source: "HuggingFaceH4/ultrafeedback_binarized"
    mix_ratio: 0.6
    format: "huggingface"
    text_field: "chosen"  # Has both chosen and rejected
    splits: ["train_prefs"]
    max_samples: 30000
    
    filters:
      - type: "length"
        min_length: 50
        max_length: 2000
  
  # 40% - RLHF preference data
  - name: "hh_rlhf_helpful"
    source: "Anthropic/hh-rlhf"
    mix_ratio: 0.4
    format: "huggingface"
    text_field: "chosen"
    splits: ["train"]
    max_samples: 20000
    
    filters:
      - type: "length"
        min_length: 30
        max_length: 2000

validation:
  ratio: 0.05
  seed: 42
```

#### 3.2 Safety DPO (`config/data/posttrain/dpo/safety.yaml`)

**Goal:** Align model for safe, harmless responses

```yaml
metadata:
  phase: "dpo_safety"
  description: "Preference optimization for safety and harmlessness"
  total_tokens: 3_000_000
  max_seq_length: 1024
  version: "1.0"

datasets:
  # 100% - Safety-focused preferences
  - name: "hh_rlhf_harmless"
    source: "Anthropic/hh-rlhf"
    mix_ratio: 1.0
    format: "huggingface"
    text_field: "chosen"
    splits: ["train"]
    max_samples: 50000
    
    filters:
      - type: "length"
        min_length: 30
        max_length: 2000
      
      - type: "quality"
        min_score: 0.2

validation:
  ratio: 0.05
  seed: 42
```

---

### **Step 4: Create RLAIF Configs** ⏰ 20 min

Create 3 RLAIF configs for online learning methods.

#### 4.1 PPO (`config/data/posttrain/rlaif/ppo.yaml`)

**Goal:** Prompts for PPO-based online learning

```yaml
metadata:
  phase: "rlaif_ppo"
  description: "Prompts for PPO training with reward model"
  total_tokens: 2_000_000
  max_seq_length: 512
  version: "1.0"

datasets:
  # 100% - Diverse prompts for online learning
  - name: "ultrachat_prompts"
    source: "HuggingFaceH4/ultrachat_200k"
    mix_ratio: 1.0
    format: "huggingface"
    text_field: "messages"  # Extract first user message as prompt
    splits: ["train_sft"]
    max_samples: 30000
    
    filters:
      - type: "length"
        min_length: 20
        max_length: 500  # Prompts are shorter

validation:
  ratio: 0.05
  seed: 42
```

#### 4.2 GRPO (`config/data/posttrain/rlaif/grpo.yaml`)

**Goal:** Group-based preference optimization prompts

```yaml
metadata:
  phase: "rlaif_grpo"
  description: "Prompts for Group Relative Policy Optimization"
  total_tokens: 2_000_000
  max_seq_length: 512
  version: "1.0"

datasets:
  # 100% - Diverse user prompts
  - name: "oasst_prompts"
    source: "OpenAssistant/oasst1"
    mix_ratio: 1.0
    format: "huggingface"
    text_field: "text"
    splits: ["train"]
    max_samples: 30000
    
    filters:
      - type: "length"
        min_length: 15
        max_length: 500

validation:
  ratio: 0.05
  seed: 42
```

#### 4.3 SPO (`config/data/posttrain/rlaif/spo.yaml`)

**Goal:** Self-play optimization prompts

```yaml
metadata:
  phase: "rlaif_spo"
  description: "Prompts for Self-Play Optimization"
  total_tokens: 1_500_000
  max_seq_length: 512
  version: "1.0"

datasets:
  # 100% - Arena-style user queries
  - name: "chatbot_arena_prompts"
    source: "lmsys/chatbot_arena_conversations"
    mix_ratio: 1.0
    format: "huggingface"
    text_field: "prompt"
    splits: ["train"]
    max_samples: 25000
    
    filters:
      - type: "length"
        min_length: 10
        max_length: 500

validation:
  ratio: 0.05
  seed: 42
```

---

### **Step 5: Update Dataset Loader** ⏰ 30 min

Need to handle format conversions in `dataset/loader.py`.

#### 5.1 Add Format Converters

Many HF datasets don't have exactly the right field names. Add converters:

```python
# In dataset/loader.py

def convert_to_conversations(sample: Dict, source_format: str) -> Dict:
    """
    Convert various dataset formats to standard 'conversations' format.
    
    Standard format:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    if source_format == "ultrachat":
        # HuggingFaceH4/ultrachat_200k format
        # Has 'messages' field: [{"role": "user", "content": "..."}, ...]
        return {"conversations": sample.get("messages", [])}
    
    elif source_format == "alpaca":
        # tatsu-lab/alpaca format
        # Has: instruction, input, output
        conversations = []
        user_msg = sample["instruction"]
        if sample.get("input"):
            user_msg += "\n" + sample["input"]
        conversations.append({"role": "user", "content": user_msg})
        conversations.append({"role": "assistant", "content": sample["output"]})
        return {"conversations": conversations}
    
    elif source_format == "openorca":
        # Open-Orca/OpenOrca format
        # Has: system_prompt, question, response
        conversations = []
        if sample.get("system_prompt"):
            conversations.append({"role": "system", "content": sample["system_prompt"]})
        conversations.append({"role": "user", "content": sample["question"]})
        conversations.append({"role": "assistant", "content": sample["response"]})
        return {"conversations": conversations}
    
    else:
        # Default: assume already in correct format
        return sample
```

#### 5.2 Update YAML Schema

Add `format_converter` field to dataset config:

```yaml
datasets:
  - name: "alpaca"
    source: "tatsu-lab/alpaca"
    format: "huggingface"
    format_converter: "alpaca"  # NEW: specifies conversion method
    text_field: "text"
    # ...
```

---

### **Step 6: Create README Files** ⏰ 20 min

Create README.md in each config directory explaining:
- Available configs
- Dataset sources
- Usage examples
- How to add new configs

#### `config/data/posttrain/sft/README.md`
#### `config/data/posttrain/dpo/README.md`
#### `config/data/posttrain/rlaif/README.md`

---

### **Step 7: Testing** ⏰ 30 min

Test each config:

```bash
# Test SFT configs load
python -c "from dataset.mixer import DatasetMixer; \
           m = DatasetMixer.from_yaml('config/data/posttrain/sft/general.yaml'); \
           print(m.validate_mixture())"

# Test DPO configs load
python -c "from dataset.mixer import DatasetMixer; \
           m = DatasetMixer.from_yaml('config/data/posttrain/dpo/helpfulness.yaml'); \
           print(m.validate_mixture())"

# Test RLAIF configs load
python -c "from dataset.mixer import DatasetMixer; \
           m = DatasetMixer.from_yaml('config/data/posttrain/rlaif/ppo.yaml'); \
           print(m.validate_mixture())"
```

---

## 📁 Final Directory Structure

```
config/data/posttrain/
├── sft/
│   ├── general.yaml      ✅ NEW
│   ├── code.yaml         ✅ NEW
│   ├── reasoning.yaml    ✅ NEW
│   └── README.md         ✅ NEW
├── dpo/
│   ├── helpfulness.yaml  ✅ NEW
│   ├── safety.yaml       ✅ NEW
│   └── README.md         ✅ NEW
└── rlaif/
    ├── ppo.yaml          ✅ NEW
    ├── grpo.yaml         ✅ NEW
    ├── spo.yaml          ✅ NEW
    └── README.md         ✅ NEW
```

---

## ⚠️ Known Challenges

### Challenge 1: Format Mismatch
**Problem:** HF datasets have different field names than our dataset classes expect.

**Solution:**
- Add `format_converter` parameter to configs
- Implement converters in `loader.py`
- Convert during dataset loading, before mixing

### Challenge 2: DPO Format Complexity
**Problem:** DPO datasets need both `chosen` and `rejected` fields.

**Solution:**
- Some datasets already have this (ultrafeedback, hh-rlhf)
- Mixer should preserve both fields when saving to JSONL
- Update `mixer.py` to handle preference pairs

### Challenge 3: Dataset Sizes
**Problem:** Some HF datasets are very large (>100GB).

**Solution:**
- Use `max_samples` to limit size
- Use streaming mode for very large datasets (future)
- Download only needed splits

### Challenge 4: Data Quality Variation
**Problem:** Different datasets have different quality levels.

**Solution:**
- Apply appropriate filters per dataset
- Adjust `min_score` based on dataset
- Can add dataset-specific filters later

---

## 🧪 Testing Strategy

### Unit Tests
```python
# Test format converters
def test_alpaca_converter():
    sample = {"instruction": "Q", "input": "", "output": "A"}
    result = convert_to_conversations(sample, "alpaca")
    assert "conversations" in result
    assert len(result["conversations"]) == 2

# Test config validation
def test_sft_config_valid():
    mixer = DatasetMixer.from_yaml("config/data/posttrain/sft/general.yaml")
    assert mixer.validate_mixture()["is_valid"]
```

### Integration Tests
```bash
# Prepare small SFT dataset
python scripts/prepare_dataset.py \
    --config config/data/posttrain/sft/general.yaml \
    --output_dir dataset/test/

# Verify JSONL format
python -c "
import json
with open('dataset/test/sft_general_train.jsonl') as f:
    sample = json.loads(f.readline())
    assert 'conversations' in sample or 'text' in sample
    print('✅ Format OK')
"
```

---

## ✅ Success Criteria

Task 2 complete when:
- [x] All 8 YAML configs created
- [x] All configs validate successfully
- [x] Format converters implemented
- [x] README files written
- [x] Can load each config without errors
- [x] Can prepare at least one dataset from each category
- [x] JSONL outputs have correct format for dataset classes

---

## 📊 Estimated Time

| Task | Time | Priority |
|------|------|----------|
| Research datasets | 15 min | HIGH |
| Create SFT configs (3) | 30 min | HIGH |
| Create DPO configs (2) | 20 min | HIGH |
| Create RLAIF configs (3) | 20 min | MEDIUM |
| Update loader.py | 30 min | HIGH |
| Create READMEs (3) | 20 min | MEDIUM |
| Testing | 30 min | HIGH |

**Total:** ~2.5 hours

---

## 🚀 Implementation Order

1. **Create directory structure** (2 min)
2. **Create SFT configs** - most important (30 min)
3. **Update loader.py with converters** (30 min)
4. **Test SFT config loading** (10 min)
5. **Create DPO configs** (20 min)
6. **Create RLAIF configs** (20 min)
7. **Create README files** (20 min)
8. **Full testing** (20 min)
9. **Documentation** (10 min)

---

## 📝 Next Steps After Task 2

Once configs are ready, we'll move to **Task 3: Enhance Post-Training Scripts** to make them use these configs properly.

---

## 🤔 Review Questions

Before starting implementation:

1. **Are the dataset choices appropriate?**
   - SFT: UltraChat, OpenOrca, Alpaca, MathInstruct
   - DPO: UltraFeedback, HH-RLHF
   - RLAIF: UltraChat, OASST, ChatArena

2. **Are mix ratios reasonable?**
   - General SFT: 50/30/20
   - Code SFT: 70/30
   - Helpfulness DPO: 60/40

3. **Should we add more configs?**
   - Domain-specific SFT (medical, legal)?
   - More DPO types (formatting, conciseness)?

4. **Is format conversion approach OK?**
   - Add converters to loader.py?
   - Or preprocess datasets beforehand?

---

**Status:** 📝 **AWAITING APPROVAL TO START TASK 2**

Please review and approve, or suggest changes!

---

**Document Version:** 1.0  
**Author:** AI Assistant  
**Last Updated:** 2025-11-15
