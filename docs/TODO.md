# MiniGPT → SOTA 7B Model Training Pipeline: TODO & Roadmap

**Final Goal:** Train a State-of-the-Art 7B parameter language model from scratch using a modular, experiment-friendly pipeline.

**Strategy:** Transform miniGPT into a flexible experimentation framework that supports:
- Multiple dataset mixtures per training phase
- Multiple model architectures (Dense, MoE, Hybrid)
- Complete training lifecycle: Pretrain → Mid-train → Post-train (SFT/RLHF/RLAIF)
- Small-scale validation before expensive large-scale training

---

## 📋 Project Phases Overview

### Phase 1: Dataset Mixture Infrastructure ✅ (Current Priority)
Build a modular dataset configuration system supporting multiple phases and mixtures.

### Phase 2: Model Architecture Modularity
Refactor model implementations to support pluggable architectures.

### Phase 3: Training Pipeline Integration
Unify training scripts to work with any dataset mixture + model architecture combination.

### Phase 4: Experiment Management & Tracking
Add tools for managing experiments, comparing results, and scaling to 7B.

---

## 🎯 Phase 1: Dataset Mixture Infrastructure

### 1.1 Create Dataset Configuration System
**Goal:** YAML-based dataset mixture definitions for each training phase.

**Directory Structure:**
```
miniGPT/
├── config/
│   └── data/
│       ├── pretrain/
│       │   ├── phase1/
│       │   │   ├── mixture1.yaml
│       │   │   ├── mixture2.yaml
│       │   │   └── mixture3.yaml
│       │   └── phase2/
│       │       ├── mixture1.yaml
│       │       └── mixture2.yaml
│       ├── midtrain/
│       │   ├── phase1/
│       │   │   ├── mixture1.yaml
│       │   │   └── mixture2.yaml
│       │   └── phase2/
│       │       └── mixture1.yaml
│       └── posttrain/
│           ├── sft/
│           │   ├── general.yaml
│           │   ├── code.yaml
│           │   └── reasoning.yaml
│           ├── dpo/
│           │   ├── helpfulness.yaml
│           │   └── safety.yaml
│           ├── ppo/
│           │   └── reward_model.yaml
│           └── rlaif/
│               └── self_improvement.yaml
```

**Tasks:**
- [ ] Create `config/data/` directory structure
- [ ] Design YAML schema for dataset mixtures (see template below)
- [ ] Create example mixture files for each phase
- [ ] Document dataset sources and access methods

**YAML Schema Template:**
```yaml
metadata:
  phase: "pretrain_phase1"
  total_tokens: 2_000_000_000_000  # 2T tokens
  steps: 500_000
  description: "Foundation phase with balanced general knowledge"
  max_seq_length: 4096
  
datasets:
  # Each dataset entry
  - name: "dataset_identifier"
    source: "huggingface/dataset-name"  # or local path
    mix_ratio: 0.30  # 30% of total tokens
    estimated_tokens: 600_000_000_000
    format: "jsonl"  # or parquet, arrow, etc.
    text_field: "content"  # field containing text
    splits: ["train"]  # which splits to use
    
    # Optional filters/preprocessing
    filters:
      - type: "language"
        languages: ["en"]
      - type: "quality"
        min_length: 100
        max_length: 100000
      - type: "deduplication"
        method: "exact"  # or "minhash", "simhash"
      - type: "pii_removal"
        enabled: true
    
    # Optional sampling strategy
    sampling:
      method: "uniform"  # or "weighted", "curriculum"
      temperature: 1.0
      
validation:
  ratio: 0.01  # 1% for validation
  seed: 42
```

### 1.2 Build Dataset Loader & Mixer
**Goal:** Python module to parse YAML configs and create mixed datasets.

**Implementation Plan:**
```
miniGPT/
├── dataset/
│   ├── mixer.py              # Main dataset mixing logic
│   ├── loader.py             # Load individual datasets
│   ├── filters.py            # Quality filters, dedup, etc.
│   ├── samplers.py           # Sampling strategies
│   └── validation.py         # Dataset validation utils
```

**Tasks:**
- [ ] Create `dataset/mixer.py` - Core mixing engine
  - [ ] Parse YAML configuration files
  - [ ] Load multiple datasets from various sources (HF, local, etc.)
  - [ ] Implement proportional sampling based on mix_ratio
  - [ ] Handle token counting and batch creation
  
- [ ] Create `dataset/loader.py` - Dataset loading utilities
  - [ ] Support HuggingFace datasets
  - [ ] Support local JSONL/Parquet files
  - [ ] Support streaming for large datasets
  - [ ] Implement caching mechanisms
  
- [ ] Create `dataset/filters.py` - Data quality filters
  - [ ] Language detection filter
  - [ ] Length-based filters (min/max tokens)
  - [ ] Quality scoring (perplexity-based, etc.)
  - [ ] PII detection and removal
  - [ ] Deduplication (exact, fuzzy, near-duplicate)
  
- [ ] Create `dataset/samplers.py` - Sampling strategies
  - [ ] Uniform sampling
  - [ ] Weighted sampling (by quality scores)
  - [ ] Curriculum learning sampling
  - [ ] Temperature-based sampling
  
- [ ] Create `dataset/validation.py` - Validation utilities
  - [ ] Verify dataset mixture ratios
  - [ ] Calculate actual vs expected token counts
  - [ ] Generate dataset statistics reports

### 1.3 Create Example Dataset Mixtures

**Tasks:**
- [ ] **Pretrain Phase 1** (2T tokens - Foundation)
  - [ ] `mixture1.yaml`: Balanced mix (web, code, math, books)
  - [ ] `mixture2.yaml`: Code-heavy mix
  - [ ] `mixture3.yaml`: Knowledge-heavy mix
  
- [ ] **Pretrain Phase 2** (500B tokens - Refinement)
  - [ ] `mixture1.yaml`: High-quality filtered data
  - [ ] `mixture2.yaml`: Domain-specific boost
  
- [ ] **Midtrain Phase 1** (100B tokens - Specialization)
  - [ ] `mixture1.yaml`: Reasoning & math
  - [ ] `mixture2.yaml`: Code & technical
  
- [ ] **Midtrain Phase 2** (50B tokens - Fine-tuning prep)
  - [ ] `mixture1.yaml`: High-quality instruction data
  
- [ ] **Post-train SFT**
  - [ ] `general.yaml`: General instruction following
  - [ ] `code.yaml`: Code generation & debugging
  - [ ] `reasoning.yaml`: Chain-of-thought reasoning
  
- [ ] **Post-train DPO**
  - [ ] `helpfulness.yaml`: Helpful vs unhelpful responses
  - [ ] `safety.yaml`: Safe vs unsafe responses
  
- [ ] **Post-train PPO/RLAIF**
  - [ ] `reward_model.yaml`: Reward model training data
  - [ ] `self_improvement.yaml`: AI feedback loops

### 1.4 Dataset Utilities & CLI

**Tasks:**
- [ ] Create `scripts/prepare_dataset.py` - CLI tool
  - [ ] Load and validate mixture configs
  - [ ] Download/cache datasets
  - [ ] Apply filters and preprocessing
  - [ ] Generate dataset statistics
  - [ ] Create train/val splits
  - [ ] Save preprocessed datasets
  
- [ ] Create `scripts/analyze_mixture.py` - Analysis tool
  - [ ] Calculate actual token distributions
  - [ ] Generate mixture reports (CSV, JSON)
  - [ ] Visualize dataset composition
  - [ ] Estimate training time and cost

---

## 🏗️ Phase 2: Model Architecture Modularity

### 2.1 Refactor Model Structure
**Goal:** HuggingFace-style modular architecture supporting multiple model types.

**Target Directory Structure:**
```
miniGPT/
├── model/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── config.py          # BaseConfig (shared)
│   │   └── modeling.py        # Base modeling utilities
│   ├── minimind/
│   │   ├── __init__.py
│   │   ├── config.py          # MiniMindConfig
│   │   ├── modeling.py        # MiniMindForCausalLM
│   │   └── modeling_moe.py    # MiniMindMoEForCausalLM
│   ├── qwen/
│   │   ├── __init__.py
│   │   ├── config.py          # QwenConfig
│   │   └── modeling.py        # QwenForCausalLM
│   ├── llama/
│   │   ├── __init__.py
│   │   ├── config.py          # LlamaConfig
│   │   └── modeling.py        # LlamaForCausalLM
│   ├── deepseek/
│   │   ├── __init__.py
│   │   ├── config.py          # DeepSeekConfig
│   │   └── modeling.py        # DeepSeekForCausalLM (MoE)
│   └── registry.py            # Model registry for easy switching
```

### 2.2 Create Base Model Interface

**Tasks:**
- [ ] Create `model/base/config.py`
  - [ ] Define `BaseModelConfig` with common hyperparameters
  - [ ] Standard parameter names across all models
  - [ ] Validation logic for config consistency
  
- [ ] Create `model/base/modeling.py`
  - [ ] `BaseModelForCausalLM` abstract class
  - [ ] Standard forward() signature
  - [ ] Standard generate() method
  - [ ] Loss computation utilities
  - [ ] Gradient checkpointing support
  
- [ ] Create `model/registry.py`
  - [ ] Model registry pattern
  - [ ] Factory functions: `create_model(model_type, config)`
  - [ ] Auto-discovery of available models

### 2.3 Refactor Existing MiniMind Model

**Tasks:**
- [ ] Move current implementation to `model/minimind/`
- [ ] Split `model_minimind.py` into:
  - [ ] `config.py` - MiniMindConfig (inherit from BaseModelConfig)
  - [ ] `modeling.py` - MiniMindForCausalLM (inherit from BaseModelForCausalLM)
  - [ ] `modeling_moe.py` - MiniMindMoEForCausalLM
- [ ] Update imports across the codebase
- [ ] Maintain backward compatibility with existing checkpoints

### 2.4 Add New Model Architectures

**Priority 1: Llama-style Architecture**
- [ ] Implement `model/llama/config.py`
  - [ ] Support for 1B, 3B, 7B, 13B parameter configs
  - [ ] GQA (Grouped Query Attention)
  - [ ] RoPE position embeddings
  - [ ] SwiGLU activation
  
- [ ] Implement `model/llama/modeling.py`
  - [ ] LlamaAttention with GQA
  - [ ] LlamaMLP with SwiGLU
  - [ ] LlamaDecoderLayer
  - [ ] LlamaForCausalLM
  - [ ] Flash Attention support

**Priority 2: Qwen-style Architecture**
- [ ] Implement `model/qwen/config.py`
  - [ ] Qwen2-specific configurations
  - [ ] Sliding window attention (optional)
  - [ ] Dynamic NTK RoPE scaling
  
- [ ] Implement `model/qwen/modeling.py`
  - [ ] QwenAttention
  - [ ] QwenMLP
  - [ ] QwenForCausalLM

**Priority 3: DeepSeek MoE Architecture**
- [ ] Implement `model/deepseek/config.py`
  - [ ] MoE-specific parameters (num_experts, experts_per_token)
  - [ ] Shared + Routed experts configuration
  - [ ] Load balancing parameters
  
- [ ] Implement `model/deepseek/modeling.py`
  - [ ] DeepSeekMoE layer
  - [ ] Expert routing logic
  - [ ] Load balancing loss
  - [ ] DeepSeekForCausalLM

### 2.5 Model Configuration Files

**Tasks:**
- [ ] Create `config/model/` directory
- [ ] Add model config YAMLs for each architecture:
  ```
  config/model/
  ├── minimind/
  │   ├── 26m.yaml
  │   ├── 104m.yaml
  │   └── 145m_moe.yaml
  ├── llama/
  │   ├── 1b.yaml
  │   ├── 3b.yaml
  │   └── 7b.yaml
  ├── qwen/
  │   ├── 1.5b.yaml
  │   └── 7b.yaml
  └── deepseek/
      ├── 7b_moe.yaml
      └── 16b_moe.yaml
  ```

**YAML Model Config Template:**
```yaml
model_type: "llama"
architecture: "decoder_only"

# Basic model dimensions
hidden_size: 4096
intermediate_size: 11008
num_hidden_layers: 32
num_attention_heads: 32
num_key_value_heads: 8  # GQA

# Vocab and embeddings
vocab_size: 32000
max_position_embeddings: 4096
rope_theta: 10000.0

# Normalization and activations
rms_norm_eps: 1.0e-6
hidden_act: "silu"

# Training
tie_word_embeddings: false
initializer_range: 0.02

# Optimization
use_flash_attention: true
gradient_checkpointing: false

# MoE (if applicable)
use_moe: false
num_experts: 8
num_experts_per_tok: 2
```

---

## 🔧 Phase 3: Training Pipeline Integration

### 3.1 Update Training Scripts

**Tasks:**
- [ ] Refactor `trainer/train_pretrain.py`
  - [ ] Accept `--data_config` argument (path to mixture YAML)
  - [ ] Accept `--model_config` argument (path to model YAML)
  - [ ] Load dataset using mixer module
  - [ ] Create model using registry
  - [ ] Keep existing training loop logic
  
- [ ] Refactor `trainer/train_full_sft.py`
  - [ ] Support SFT mixture configs
  - [ ] Model-agnostic training
  
- [ ] Refactor `trainer/train_dpo.py`
  - [ ] Support DPO mixture configs
  - [ ] Model-agnostic training
  
- [ ] Update all other training scripts similarly:
  - [ ] `train_lora.py`
  - [ ] `train_ppo.py`
  - [ ] `train_grpo.py`
  - [ ] `train_spo.py`
  - [ ] `train_distillation.py`

### 3.2 Create Unified Training CLI

**Tasks:**
- [ ] Create `scripts/train.py` - Universal training entry point
  ```bash
  python scripts/train.py \
    --stage pretrain \
    --phase phase1 \
    --data_mixture mixture1 \
    --model_config llama/7b \
    --output_dir experiments/exp001 \
    --run_name "llama-7b-pretrain-phase1-mix1"
  ```
  
- [ ] Support all training stages from one interface
- [ ] Automatic experiment directory creation
- [ ] Config validation before training starts

### 3.3 Experiment Configuration System

**Tasks:**
- [ ] Create `config/experiments/` directory
- [ ] Add experiment templates:
  ```
  config/experiments/
  ├── pretrain_template.yaml
  ├── sft_template.yaml
  ├── dpo_template.yaml
  └── full_pipeline.yaml
  ```

**Experiment Config Template:**
```yaml
experiment:
  name: "llama-7b-pretrain-phase1"
  description: "First pretraining phase with balanced mixture"
  tags: ["pretrain", "llama", "7b", "phase1"]

model:
  config_path: "config/model/llama/7b.yaml"
  from_scratch: true
  # Optional: load from checkpoint
  # checkpoint_path: "path/to/checkpoint.pth"

data:
  config_path: "config/data/pretrain/phase1/mixture1.yaml"
  num_workers: 8
  prefetch_factor: 2

training:
  # Training hyperparameters
  batch_size: 512
  gradient_accumulation_steps: 16
  max_steps: 500_000
  learning_rate: 3.0e-4
  warmup_steps: 2000
  lr_scheduler: "cosine"
  weight_decay: 0.1
  grad_clip: 1.0
  
  # Optimization
  optimizer: "adamw"
  betas: [0.9, 0.95]
  eps: 1.0e-8
  
  # Mixed precision
  mixed_precision: "bf16"
  
  # Distributed training
  use_ddp: true
  world_size: 8

checkpointing:
  save_steps: 1000
  save_total_limit: 5
  output_dir: "experiments/llama-7b-pretrain-phase1"

logging:
  log_steps: 10
  eval_steps: 500
  use_wandb: true
  wandb_project: "minimind-7b"
  wandb_run_name: "llama-7b-pretrain-phase1-mix1"
```

### 3.4 Add Resume & Checkpoint Management

**Tasks:**
- [ ] Enhance `trainer_utils.py` checkpoint functions
  - [ ] Save optimizer state, scheduler state, RNG states
  - [ ] Save dataset state (resume from exact position)
  - [ ] Automatic checkpoint cleanup
  
- [ ] Add `scripts/resume_training.py`
  - [ ] Auto-detect latest checkpoint
  - [ ] Validate checkpoint compatibility
  - [ ] Resume training seamlessly

---

## 📊 Phase 4: Experiment Management & Tracking

### 4.1 Experiment Tracking Integration

**Tasks:**
- [ ] Enhance Weights & Biases (wandb) integration
  - [ ] Log dataset mixture details
  - [ ] Log model architecture details
  - [ ] Track training metrics (loss, perplexity, throughput)
  - [ ] Log system metrics (GPU util, memory, etc.)
  
- [ ] Add TensorBoard support as alternative
  
- [ ] Create experiment comparison dashboard
  - [ ] Compare multiple runs side-by-side
  - [ ] Visualize loss curves
  - [ ] Generate comparison reports

### 4.2 Evaluation Framework

**Tasks:**
- [ ] Create `evaluation/` directory
  ```
  evaluation/
  ├── benchmarks/
  │   ├── mmlu.py
  │   ├── hellaswag.py
  │   ├── humaneval.py
  │   └── gsm8k.py
  ├── metrics/
  │   ├── perplexity.py
  │   ├── accuracy.py
  │   └── generation_quality.py
  └── runner.py
  ```
  
- [ ] Integrate standard LLM benchmarks:
  - [ ] MMLU (Massive Multitask Language Understanding)
  - [ ] HellaSwag (Commonsense reasoning)
  - [ ] HumanEval (Code generation)
  - [ ] GSM8K (Math reasoning)
  - [ ] TruthfulQA (Truthfulness)
  - [ ] BigBench (Diverse tasks)
  
- [ ] Create `scripts/evaluate.py` CLI tool
  ```bash
  python scripts/evaluate.py \
    --checkpoint experiments/exp001/checkpoint-final \
    --benchmarks mmlu,hellaswag,humaneval \
    --output_dir evaluation_results/exp001
  ```

### 4.3 Analysis & Visualization Tools

**Tasks:**
- [ ] Create `scripts/analyze_training.py`
  - [ ] Parse training logs
  - [ ] Generate loss curve plots
  - [ ] Calculate training efficiency metrics
  - [ ] Estimate cost and carbon footprint
  
- [ ] Create `scripts/compare_experiments.py`
  - [ ] Compare multiple experiments
  - [ ] Generate comparison tables (markdown/CSV)
  - [ ] Statistical significance tests
  
- [ ] Create `scripts/visualize_attention.py`
  - [ ] Attention pattern visualization
  - [ ] Token attribution analysis
  - [ ] Layer-wise analysis tools

### 4.4 Documentation & Guides

**Tasks:**
- [ ] Create comprehensive documentation:
  - [ ] `docs/DATASET_MIXING_GUIDE.md`
  - [ ] `docs/MODEL_ARCHITECTURE_GUIDE.md`
  - [ ] `docs/TRAINING_GUIDE.md`
  - [ ] `docs/EVALUATION_GUIDE.md`
  - [ ] `docs/SCALING_TO_7B.md`
  
- [ ] Add tutorial notebooks:
  - [ ] `notebooks/01_dataset_preparation.ipynb`
  - [ ] `notebooks/02_model_training.ipynb`
  - [ ] `notebooks/03_evaluation.ipynb`
  
- [ ] Create experiment recipes:
  - [ ] `recipes/pretrain_1b.md`
  - [ ] `recipes/pretrain_7b.md`
  - [ ] `recipes/full_pipeline.md`

---

## 🚀 Phase 5: Scaling to 7B (Future)

### 5.1 Infrastructure Requirements

**Tasks:**
- [ ] Calculate compute requirements for 7B training
  - [ ] Estimate FLOPs, GPU hours, memory requirements
  - [ ] Plan distributed training strategy (DDP, FSDP, DeepSpeed)
  
- [ ] Set up multi-node training infrastructure
  - [ ] Configure distributed training backend
  - [ ] Test communication bandwidth
  - [ ] Optimize data loading pipeline
  
- [ ] Implement advanced optimization techniques:
  - [ ] Flash Attention 2
  - [ ] Gradient checkpointing
  - [ ] Mixed precision (BF16/FP8)
  - [ ] ZeRO optimizer stages

### 5.2 Dataset Preparation

**Tasks:**
- [ ] Acquire/download large-scale datasets
  - [ ] RedPajama, Pile, C4, Common Crawl
  - [ ] StarCoder (code)
  - [ ] OpenWebMath (math)
  - [ ] Wikipedia, Books3 (knowledge)
  
- [ ] Run preprocessing pipeline at scale
  - [ ] Quality filtering (perplexity-based)
  - [ ] Deduplication (exact + fuzzy)
  - [ ] PII removal
  - [ ] Language detection
  
- [ ] Create final dataset mixtures (2T+ tokens)

### 5.3 Training Execution

**Tasks:**
- [ ] Run Phase 1 pretraining (2T tokens)
- [ ] Run Phase 2 pretraining (500B tokens)
- [ ] Run midtraining (150B tokens)
- [ ] Run SFT (10B tokens)
- [ ] Run RLHF/DPO
- [ ] Continuous evaluation and iteration

### 5.4 Model Release

**Tasks:**
- [ ] Prepare model for release
  - [ ] Convert to HuggingFace format
  - [ ] Create model card with training details
  - [ ] Benchmark on standard evals
  
- [ ] Release checkpoints
  - [ ] Upload to HuggingFace Hub
  - [ ] Upload to ModelScope
  
- [ ] Write technical report/blog post

---

## 📅 Implementation Timeline Estimate

### Sprint 1-2 (Weeks 1-2): Dataset Infrastructure
- Complete Phase 1.1 - 1.4
- Test with small-scale examples

### Sprint 3-4 (Weeks 3-4): Model Architecture Refactor
- Complete Phase 2.1 - 2.3
- Add Llama architecture (Priority 1)

### Sprint 5-6 (Weeks 5-6): Training Pipeline Integration
- Complete Phase 3.1 - 3.4
- End-to-end testing with multiple models

### Sprint 7-8 (Weeks 7-8): Experiment Management
- Complete Phase 4.1 - 4.4
- Run small-scale experiments (100M-500M params)

### Sprint 9+ (Week 9+): Scaling & Iteration
- Iterate on mixtures and architectures
- Scale to 1B → 3B → 7B progressively

---

## ✅ Success Criteria

### Phase 1 Success Metrics:
- [ ] Can define dataset mixture in YAML
- [ ] Can load and mix multiple datasets
- [ ] Mixture ratios are correctly maintained
- [ ] Preprocessing filters work as expected

### Phase 2 Success Metrics:
- [ ] Can switch between model architectures with config change
- [ ] All models implement same interface
- [ ] Backward compatibility with existing checkpoints
- [ ] Can train Llama-style model from scratch

### Phase 3 Success Metrics:
- [ ] Single command trains any model + mixture combination
- [ ] Training can resume from checkpoints
- [ ] All stages (pretrain/sft/dpo) work with new system

### Phase 4 Success Metrics:
- [ ] Experiments are tracked and comparable
- [ ] Can evaluate on standard benchmarks
- [ ] Clear winner selection from multiple experiments

### Phase 5 Success Metrics:
- [ ] 7B model successfully trained
- [ ] Competitive benchmark results
- [ ] Total cost < target budget
- [ ] Model released publicly

---

## 🛠️ Technical Debt & Considerations

### Must Address:
- [ ] Memory optimization for large-scale training
- [ ] Efficient data loading (avoid bottlenecks)
- [ ] Checkpoint size management (sharding, quantization)
- [ ] Error handling and recovery mechanisms
- [ ] Comprehensive testing (unit + integration)

### Nice to Have:
- [ ] Web UI for experiment management
- [ ] Automatic hyperparameter tuning
- [ ] Model quantization pipeline (INT8, INT4)
- [ ] ONNX/TensorRT export support
- [ ] Multi-language tokenizer experiments

---

## 📚 References & Resources

### Dataset Sources:
- **General Text:** RedPajama, The Pile, C4, Common Crawl
- **Code:** StarCoder, The Stack, GitHub Code
- **Math:** OpenWebMath, MATH dataset, GSM8K
- **Books:** Books3, Gutenberg, BookCorpus
- **Scientific:** ArXiv, PubMed, S2ORC

### Model Architecture Papers:
- **Llama 3.1:** [Meta's Technical Report]
- **Qwen2:** [Alibaba's Technical Report]
- **DeepSeek V2/V3:** [DeepSeek MoE Architecture]
- **Mixture of Depths:** [Dynamic depth allocation]

### Training Best Practices:
- **Chinchilla Scaling Laws:** Optimal tokens per parameter
- **Data Quality:** Importance of cleaning and filtering
- **Learning Rate Schedules:** Cosine with warmup
- **Curriculum Learning:** Easy to hard data ordering

---

## 🎯 Next Immediate Actions

1. **Create directory structure** for Phase 1
2. **Design and document YAML schema** for mixtures
3. **Implement basic dataset mixer** (load + combine)
4. **Create 2-3 example mixtures** for testing
5. **Write unit tests** for mixer functionality
6. **Run end-to-end test** with MiniMind on toy mixture

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-12  
**Status:** Planning Phase  
**Next Review:** After Phase 1 completion
