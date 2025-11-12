# MiniGPT Modular Training Pipeline: Architecture Design

**Version:** 1.0  
**Date:** 2025-11-12  
**Status:** Design Phase

---

## 🎯 Executive Summary

This document outlines the architectural design for transforming miniGPT from a single-model educational project into a **modular, production-grade experimentation framework** capable of training SOTA 7B language models.

**Core Principles:**
1. **Modularity:** Swap datasets and models without touching training code
2. **Reproducibility:** All experiments defined by version-controlled configs
3. **Scalability:** Design works from 100M to 7B+ parameters
4. **Efficiency:** Minimize wasted compute on failed experiments

---

## 📐 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MiniGPT Training Framework                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐
│   Dataset Configs    │     │   Model Configs      │
│   (YAML Files)       │     │   (YAML Files)       │
├──────────────────────┤     ├──────────────────────┤
│ • pretrain/phase1/   │     │ • llama/7b.yaml      │
│   - mixture1.yaml    │     │ • qwen/7b.yaml       │
│   - mixture2.yaml    │     │ • deepseek/7b.yaml   │
│ • sft/general.yaml   │     │ • minimind/104m.yaml │
└──────────────────────┘     └──────────────────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
         ┌────────────────────────┐
         │  Experiment Config     │
         │  (YAML)                │
         │  • Defines model       │
         │  • Defines data        │
         │  • Training params     │
         └────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  Dataset Mixer   │    │  Model Registry  │
│  • Load datasets │    │  • Create model  │
│  • Apply filters │    │  • Initialize    │
│  • Mix ratios    │    │  • Load ckpt     │
└──────────────────┘    └──────────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
         ┌────────────────────────┐
         │  Unified Training      │
         │  Pipeline              │
         │  • Pretrain            │
         │  • SFT                 │
         │  • DPO/PPO             │
         └────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  Checkpoints     │    │  Experiment      │
│  • Model weights │    │  Tracking        │
│  • Optimizer     │    │  • WandB/TB      │
│  • Scheduler     │    │  • Metrics       │
└──────────────────┘    └──────────────────┘
```

---

## 🗂️ Directory Structure (Final State)

```
miniGPT/
├── config/                              # All configuration files
│   ├── data/                            # Dataset mixture configs
│   │   ├── pretrain/
│   │   │   ├── phase1/
│   │   │   │   ├── mixture1.yaml       # Balanced general mix
│   │   │   │   ├── mixture2.yaml       # Code-heavy mix
│   │   │   │   └── mixture3.yaml       # Math-heavy mix
│   │   │   └── phase2/
│   │   │       └── mixture1.yaml       # Refined high-quality
│   │   ├── midtrain/
│   │   │   ├── phase1/
│   │   │   │   ├── mixture1.yaml       # Reasoning focus
│   │   │   │   └── mixture2.yaml       # Domain specialization
│   │   │   └── phase2/
│   │   │       └── mixture1.yaml       # Pre-SFT preparation
│   │   └── posttrain/
│   │       ├── sft/
│   │       │   ├── general.yaml        # General instruction
│   │       │   ├── code.yaml           # Code generation
│   │       │   └── reasoning.yaml      # CoT reasoning
│   │       ├── dpo/
│   │       │   ├── helpfulness.yaml
│   │       │   └── safety.yaml
│   │       └── ppo/
│   │           └── reward.yaml
│   ├── model/                           # Model architecture configs
│   │   ├── llama/
│   │   │   ├── 1b.yaml
│   │   │   ├── 3b.yaml
│   │   │   └── 7b.yaml
│   │   ├── qwen/
│   │   │   ├── 1.5b.yaml
│   │   │   └── 7b.yaml
│   │   ├── deepseek/
│   │   │   └── 7b_moe.yaml
│   │   └── minimind/
│   │       ├── 26m.yaml
│   │       └── 104m.yaml
│   └── experiments/                     # Full experiment configs
│       ├── templates/
│       │   ├── pretrain_template.yaml
│       │   ├── sft_template.yaml
│       │   └── dpo_template.yaml
│       └── active/
│           ├── exp001_llama7b_pretrain.yaml
│           └── exp002_qwen7b_sft.yaml
│
├── model/                               # Model implementations
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── config.py                   # BaseModelConfig
│   │   └── modeling.py                 # BaseModelForCausalLM
│   ├── minimind/
│   │   ├── __init__.py
│   │   ├── config.py                   # MiniMindConfig
│   │   ├── modeling.py                 # MiniMindForCausalLM
│   │   └── modeling_moe.py             # MiniMindMoEForCausalLM
│   ├── llama/
│   │   ├── __init__.py
│   │   ├── config.py                   # LlamaConfig
│   │   └── modeling.py                 # LlamaForCausalLM
│   ├── qwen/
│   │   ├── __init__.py
│   │   ├── config.py                   # QwenConfig
│   │   └── modeling.py                 # QwenForCausalLM
│   ├── deepseek/
│   │   ├── __init__.py
│   │   ├── config.py                   # DeepSeekConfig
│   │   └── modeling.py                 # DeepSeekForCausalLM
│   ├── registry.py                      # Model factory
│   └── tokenizer/                       # Tokenizer files
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── dataset/                             # Dataset processing
│   ├── __init__.py
│   ├── lm_dataset.py                   # Original dataset classes
│   ├── mixer.py                        # NEW: Dataset mixing engine
│   ├── loader.py                       # NEW: Multi-source loading
│   ├── filters.py                      # NEW: Quality filters
│   ├── samplers.py                     # NEW: Sampling strategies
│   ├── validation.py                   # NEW: Dataset validation
│   └── dataset.md                      # Documentation
│
├── trainer/                             # Training scripts
│   ├── __init__.py
│   ├── train_pretrain.py               # UPDATED: Uses mixer + registry
│   ├── train_full_sft.py               # UPDATED: Uses mixer + registry
│   ├── train_dpo.py                    # UPDATED: Uses mixer + registry
│   ├── train_lora.py
│   ├── train_ppo.py
│   ├── train_grpo.py
│   ├── train_spo.py
│   ├── train_distillation.py
│   └── trainer_utils.py                # UPDATED: Enhanced checkpointing
│
├── scripts/                             # Utility scripts
│   ├── train.py                        # NEW: Unified training CLI
│   ├── prepare_dataset.py              # NEW: Dataset preparation
│   ├── analyze_mixture.py              # NEW: Mixture analysis
│   ├── evaluate.py                     # NEW: Model evaluation
│   ├── analyze_training.py             # NEW: Training analysis
│   ├── compare_experiments.py          # NEW: Experiment comparison
│   ├── resume_training.py              # NEW: Resume helper
│   ├── serve_openai_api.py
│   ├── chat_openai_api.py
│   ├── web_demo.py
│   └── train_tokenizer.py
│
├── evaluation/                          # NEW: Evaluation framework
│   ├── __init__.py
│   ├── benchmarks/
│   │   ├── mmlu.py
│   │   ├── hellaswag.py
│   │   ├── humaneval.py
│   │   └── gsm8k.py
│   ├── metrics/
│   │   ├── perplexity.py
│   │   └── accuracy.py
│   └── runner.py
│
├── experiments/                         # NEW: Experiment outputs
│   ├── exp001_llama7b_pretrain/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── config.yaml
│   │   └── results.json
│   └── exp002_qwen7b_sft/
│       └── ...
│
├── docs/                                # Documentation
│   ├── REPOSITORY_ANALYSIS.md
│   ├── TODO.md                         # THIS FILE
│   ├── ARCHITECTURE_PLAN.md            # This document
│   ├── DATASET_MIXING_GUIDE.md         # To be created
│   ├── MODEL_ARCHITECTURE_GUIDE.md     # To be created
│   ├── TRAINING_GUIDE.md               # To be created
│   └── SCALING_TO_7B.md                # To be created
│
├── notebooks/                           # NEW: Tutorial notebooks
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                               # NEW: Unit tests
│   ├── test_mixer.py
│   ├── test_models.py
│   └── test_training.py
│
├── requirements.txt
├── README_en.md
└── LICENSE
```

---

## 🧩 Component Design Details

### 1. Dataset Mixer Architecture

```python
# Conceptual API design

from dataset.mixer import DatasetMixer

# Load mixture configuration
mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/mixture1.yaml")

# Build the mixed dataset
mixed_dataset = mixer.build_dataset(
    tokenizer=tokenizer,
    max_seq_length=4096,
    split="train"
)

# Returns a PyTorch Dataset that yields mixed samples
for batch in DataLoader(mixed_dataset, batch_size=32):
    # Training loop
    pass
```

**Key Classes:**

```python
class DatasetMixer:
    """Main class for mixing multiple datasets according to config."""
    
    def __init__(self, config: MixtureConfig):
        self.config = config
        self.datasets = []
        self.samplers = []
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DatasetMixer':
        """Load mixer from YAML config."""
        pass
    
    def build_dataset(self, tokenizer, max_seq_length, split) -> MixedDataset:
        """Build the mixed PyTorch dataset."""
        pass
    
    def validate_mixture(self) -> Dict[str, Any]:
        """Validate that mixture ratios are achievable."""
        pass

class MixedDataset(Dataset):
    """PyTorch Dataset that samples from multiple datasets proportionally."""
    
    def __init__(self, datasets: List[Dataset], ratios: List[float]):
        self.datasets = datasets
        self.ratios = ratios
        self._build_sampling_table()
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Sample from appropriate dataset based on ratio
        pass
    
    def __len__(self) -> int:
        # Total samples based on epoch definition
        pass
```

### 2. Model Registry Architecture

```python
# Conceptual API design

from model.registry import ModelRegistry

# Register models (auto-discovered or manual)
ModelRegistry.register("llama", LlamaForCausalLM, LlamaConfig)
ModelRegistry.register("qwen", QwenForCausalLM, QwenConfig)

# Load model from config
model_config = ModelRegistry.load_config("config/model/llama/7b.yaml")
model = ModelRegistry.create_model("llama", model_config)

# Or unified interface
model = ModelRegistry.create_from_yaml("config/model/llama/7b.yaml")
```

**Key Classes:**

```python
class ModelRegistry:
    """Central registry for all model architectures."""
    
    _models = {}  # {model_type: (ModelClass, ConfigClass)}
    
    @classmethod
    def register(cls, model_type: str, model_class, config_class):
        """Register a new model architecture."""
        cls._models[model_type] = (model_class, config_class)
    
    @classmethod
    def create_model(cls, model_type: str, config) -> BaseModelForCausalLM:
        """Create model instance from type and config."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        model_class, _ = cls._models[model_type]
        return model_class(config)
    
    @classmethod
    def load_config(cls, yaml_path: str):
        """Load model config from YAML."""
        pass
    
    @classmethod
    def create_from_yaml(cls, yaml_path: str) -> BaseModelForCausalLM:
        """One-shot creation from YAML config."""
        config = cls.load_config(yaml_path)
        return cls.create_model(config.model_type, config)

class BaseModelForCausalLM(nn.Module):
    """Abstract base class for all causal language models."""
    
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Standard forward pass interface."""
        raise NotImplementedError
    
    def generate(self, input_ids, max_length=100, **kwargs):
        """Standard generation interface."""
        raise NotImplementedError
    
    def get_num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
```

### 3. Experiment Configuration Architecture

```yaml
# Full experiment definition in one file
# File: config/experiments/active/exp001_llama7b_pretrain.yaml

experiment:
  name: "exp001_llama7b_pretrain_phase1"
  description: "Llama 7B pretraining phase 1 with balanced mixture"
  tags: ["pretrain", "llama", "7b", "phase1"]
  seed: 42

# Model configuration (inline or reference)
model:
  type: "llama"
  config_path: "config/model/llama/7b.yaml"
  from_scratch: true
  # Optional overrides
  overrides:
    use_flash_attention: true
    gradient_checkpointing: true

# Dataset configuration (inline or reference)
data:
  config_path: "config/data/pretrain/phase1/mixture1.yaml"
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true

# Training hyperparameters
training:
  # Batch configuration
  batch_size: 2  # Per GPU
  gradient_accumulation_steps: 256  # Effective batch = 512 * 8 GPUs = 4096
  max_steps: 500_000
  
  # Optimization
  optimizer: "adamw"
  learning_rate: 3.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
  grad_clip: 1.0
  
  # Learning rate schedule
  lr_scheduler: "cosine"
  warmup_steps: 2000
  min_lr_ratio: 0.1
  
  # Mixed precision
  mixed_precision: "bf16"
  
  # Distributed training
  use_ddp: true
  world_size: 8
  backend: "nccl"

# Checkpointing
checkpointing:
  save_steps: 1000
  save_total_limit: 5
  output_dir: "experiments/exp001_llama7b_pretrain"
  resume_from_checkpoint: null  # or path to checkpoint

# Logging and monitoring
logging:
  log_steps: 10
  eval_steps: 500
  
  # Weights & Biases
  use_wandb: true
  wandb_project: "minimind-7b"
  wandb_entity: "your-team"
  wandb_run_name: "exp001-llama7b-pretrain-phase1"
  
  # TensorBoard (alternative)
  use_tensorboard: false
  tensorboard_dir: "experiments/exp001_llama7b_pretrain/tensorboard"

# Evaluation during training
evaluation:
  datasets:
    - name: "validation_set"
      path: "path/to/val"
  metrics:
    - "perplexity"
    - "loss"
```

### 4. Unified Training CLI

```python
# scripts/train.py - Conceptual structure

import argparse
from pathlib import Path
from dataset.mixer import DatasetMixer
from model.registry import ModelRegistry
from trainer.pretrain_trainer import PretrainTrainer
from trainer.sft_trainer import SFTTrainer
from trainer.dpo_trainer import DPOTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    # Load full experiment config
    config = ExperimentConfig.from_yaml(args.experiment_config)
    
    # Setup distributed training
    if config.training.use_ddp:
        setup_distributed(args.local_rank)
    
    # Build dataset
    mixer = DatasetMixer.from_yaml(config.data.config_path)
    dataset = mixer.build_dataset(...)
    
    # Build model
    model = ModelRegistry.create_from_yaml(config.model.config_path)
    
    # Select trainer based on stage
    stage = config.experiment.stage  # pretrain, sft, dpo, etc.
    trainer_class = {
        "pretrain": PretrainTrainer,
        "sft": SFTTrainer,
        "dpo": DPOTrainer,
    }[stage]
    
    # Initialize trainer
    trainer = trainer_class(
        model=model,
        dataset=dataset,
        config=config
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

---

## 🔄 Data Flow Diagram

```
┌─────────────┐
│ User runs:  │
│ python      │
│ scripts/    │
│ train.py    │
│ --exp=...   │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Load Experiment  │
│ Config (YAML)    │
└──────┬───────────┘
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Load Model   │   │ Load Data    │   │ Load Train   │
│ Config       │   │ Config       │   │ Config       │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       ▼                  ▼                  │
┌──────────────┐   ┌──────────────┐         │
│ Model        │   │ Dataset      │         │
│ Registry     │   │ Mixer        │         │
│ .create()    │   │ .build()     │         │
└──────┬───────┘   └──────┬───────┘         │
       │                  │                  │
       └──────────┬───────┘                  │
                  ▼                          │
         ┌────────────────┐                  │
         │ Trainer        │◄─────────────────┘
         │ (Pretrain/SFT) │
         └────────┬───────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────────┐   ┌─────────────────┐
│ Training Loop   │   │ Logging/Saving  │
│ • Forward       │   │ • Checkpoints   │
│ • Backward      │   │ • WandB logs    │
│ • Optimizer     │   │ • Eval metrics  │
└─────────────────┘   └─────────────────┘
```

---

## 🧪 Testing Strategy

### Unit Tests
```python
# tests/test_mixer.py
def test_load_mixture_config():
    mixer = DatasetMixer.from_yaml("config/data/pretrain/phase1/mixture1.yaml")
    assert mixer.config.total_tokens == 2_000_000_000_000

def test_mixture_ratios():
    mixer = DatasetMixer.from_yaml(...)
    dataset = mixer.build_dataset(...)
    # Verify actual ratios match expected
    actual_ratios = mixer.validate_mixture()
    assert abs(actual_ratios["starcoder"] - 0.1066) < 0.01

# tests/test_models.py
def test_model_creation():
    model = ModelRegistry.create_from_yaml("config/model/llama/7b.yaml")
    assert model.get_num_params() == pytest.approx(7e9, rel=0.1)

def test_model_forward():
    model = ModelRegistry.create_from_yaml("config/model/llama/1b.yaml")
    x = torch.randint(0, 32000, (2, 128))
    output = model(x)
    assert output.logits.shape == (2, 128, 32000)
```

### Integration Tests
```python
# tests/test_training.py
def test_end_to_end_training():
    """Test full training pipeline on tiny dataset."""
    # Use minimal config (10M params, 1M tokens)
    trainer = setup_trainer("tests/configs/tiny_experiment.yaml")
    trainer.train(max_steps=10)
    assert Path(trainer.config.output_dir).exists()
```

---

## 📊 Performance Optimization Strategies

### 1. Data Loading
- **Multiple workers:** `num_workers=8-16`
- **Prefetching:** `prefetch_factor=2-4`
- **Pin memory:** For faster GPU transfers
- **Persistent workers:** Avoid process restart overhead

### 2. Model Training
- **Flash Attention 2:** 2-4x speedup on attention
- **Gradient checkpointing:** Trade compute for memory
- **Mixed precision (BF16):** 2x throughput on modern GPUs
- **Gradient accumulation:** Large effective batch sizes
- **Compiled models:** `torch.compile()` for 10-20% speedup

### 3. Distributed Training
- **DDP (Distributed Data Parallel):** For multi-GPU single-node
- **FSDP (Fully Sharded Data Parallel):** For 7B+ models
- **Pipeline Parallelism:** For extremely large models
- **Tensor Parallelism:** For wide models (large hidden dim)

### 4. Checkpointing
- **Sharded checkpoints:** Save/load in parallel
- **Async saving:** Don't block training
- **Incremental checkpoints:** Only save optimizer state periodically

---

## 🚨 Critical Design Decisions

### Decision 1: Dataset Sampling Strategy
**Options:**
- A) Pre-mix datasets offline → Fixed epoch
- B) Sample on-the-fly → Dynamic mixing

**Choice:** **B - On-the-fly sampling**
**Rationale:** More flexible, supports infinite data streaming, easy ratio adjustments

### Decision 2: Model Weight Format
**Options:**
- A) Native PyTorch (.pth)
- B) HuggingFace format (safetensors)
- C) Both

**Choice:** **C - Support both**
**Rationale:** PyTorch for training checkpoints, HF for final release

### Decision 3: Config Language
**Options:**
- A) Python dataclasses
- B) YAML files
- C) JSON files

**Choice:** **B - YAML files**
**Rationale:** Human-readable, supports comments, version-controllable

### Decision 4: Tokenizer Strategy
**Options:**
- A) Shared tokenizer for all models
- B) Model-specific tokenizers

**Choice:** **A - Shared tokenizer (initially)**
**Rationale:** Simplifies dataset preparation, can change later if needed

---

## 📈 Scalability Roadmap

### Stage 1: Proof of Concept (Current)
- **Model Size:** 26M - 104M params
- **Dataset:** 1-10GB
- **Hardware:** Single GPU (3090/4090)
- **Goal:** Validate modular architecture

### Stage 2: Small Scale Validation
- **Model Size:** 500M - 1B params
- **Dataset:** 50-100GB (5-10B tokens)
- **Hardware:** 1-4 GPUs
- **Goal:** Test full pipeline, find issues

### Stage 3: Medium Scale
- **Model Size:** 3B params
- **Dataset:** 500GB-1TB (50-100B tokens)
- **Hardware:** 8 GPUs (A100/H100)
- **Goal:** Optimize efficiency, test FSDP

### Stage 4: Full Scale (Target)
- **Model Size:** 7B params
- **Dataset:** 5-10TB (500B-2T tokens)
- **Hardware:** 16-64 GPUs across nodes
- **Goal:** Train SOTA model

---

## 🎯 Key Success Metrics

### System Metrics:
- [ ] **Modularity:** Swap model in <5 minutes
- [ ] **Reproducibility:** Same config → Same result
- [ ] **Efficiency:** >50% GPU utilization
- [ ] **Reliability:** <1% job failure rate

### Model Metrics:
- [ ] **Scale:** Successfully train 7B model
- [ ] **Quality:** Competitive with open-source baselines
- [ ] **Cost:** <$10k total training cost
- [ ] **Speed:** Complete Phase 1 pretrain in <1 week

---

## 🔐 Risk Mitigation

### Risk 1: Dataset Quality Issues
- **Mitigation:** Thorough filtering, validation, human inspection
- **Fallback:** Use well-known public datasets initially

### Risk 2: OOM (Out of Memory) Errors
- **Mitigation:** Gradient checkpointing, FSDP, smaller batch sizes
- **Fallback:** Reduce model size, increase accumulation steps

### Risk 3: Training Instability
- **Mitigation:** Gradient clipping, careful LR tuning, warmup
- **Fallback:** Resume from last stable checkpoint

### Risk 4: Infrastructure Failures
- **Mitigation:** Auto-resume, redundant checkpoints, monitoring
- **Fallback:** Checkpoint every 1000 steps, keep last 5

---

## 📚 Implementation Guidelines

### Code Style
- **Docstrings:** Google style for all public functions
- **Type hints:** Use throughout
- **Comments:** Explain "why", not "what"
- **Naming:** Clear, descriptive variable names

### Git Workflow
- **Branches:** `feature/dataset-mixer`, `feature/llama-model`
- **Commits:** Small, atomic, descriptive messages
- **PRs:** Self-review before requesting review

### Documentation
- **Update TODO.md:** Mark completed items
- **Write guides:** After each phase completion
- **Add examples:** Working code snippets in docs

---

## 🎉 Conclusion

This architecture transforms miniGPT from an educational toy into a **serious LLM training framework**. The modular design enables:

1. **Rapid Experimentation:** Test 10 mixtures in time it took to test 1
2. **Reproducibility:** Version-controlled configs eliminate guesswork
3. **Scalability:** Same code works for 100M → 7B models
4. **Maintainability:** Clean abstractions make debugging easier

**Next Steps:**
1. Implement Phase 1 (Dataset Infrastructure)
2. Test with existing miniMind model
3. Add Llama architecture
4. Scale progressively: 100M → 500M → 1B → 3B → 7B

---

**Document Version:** 1.0  
**Author:** AI Assistant  
**Review Status:** Draft  
**Next Review:** After Phase 1 Implementation
