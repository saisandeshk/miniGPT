# MiniGPT (originally MiniMind) Repository Analysis

**Analysis Date:** November 9, 2025  
**Repository:** miniGPT (formerly minimind)  
**Original Author:** Jingyao Gong (@jingyaogong)

---

## 📋 Executive Summary

MiniGPT is an educational open-source project that demonstrates how to train a large language model (LLM) from scratch with minimal resources. The project showcases the complete pipeline from tokenization to reinforcement learning, specifically designed to be accessible for learning and experimentation.

**Key Highlights:**
- **Smallest Model Size:** 25.8M parameters (0.026B) - approximately 1/7000th the size of GPT-3
- **Training Cost:** ~$3 USD (2 hours on single NVIDIA 3090)
- **Architecture:** Transformer Decoder-Only (similar to Llama 3.1)
- **Full Pipeline:** Includes Pretrain, SFT, LoRA, DPO, RLAIF (PPO/GRPO/SPO), and model distillation
- **Educational Focus:** All core algorithms implemented from scratch in PyTorch without heavy dependencies

---

## 🏗️ Project Structure

```
miniGPT/
├── model/                      # Core model architecture
│   ├── model_minimind.py      # Main MiniMind model implementation
│   ├── model_lora.py          # LoRA fine-tuning implementation
│   └── tokenizer files        # Custom tokenizer (6,400 vocab size)
│
├── trainer/                    # Training scripts for different stages
│   ├── train_pretrain.py      # Pre-training stage
│   ├── train_full_sft.py      # Supervised fine-tuning (full parameters)
│   ├── train_lora.py          # LoRA fine-tuning
│   ├── train_dpo.py           # Direct Preference Optimization (RLHF)
│   ├── train_ppo.py           # Proximal Policy Optimization (RLAIF)
│   ├── train_grpo.py          # Group Relative Policy Optimization
│   ├── train_spo.py           # Self-Play Optimization
│   ├── train_distillation.py  # Model distillation
│   └── train_distill_reason.py # Reasoning model distillation
│
├── dataset/                    # Data processing and datasets
│   ├── lm_dataset.py          # Dataset loading utilities
│   └── dataset.md             # Dataset documentation
│
├── scripts/                    # Utility scripts
│   ├── train_tokenizer.py     # Train custom tokenizer
│   ├── convert_model.py       # Convert between torch/transformers formats
│   ├── serve_openai_api.py    # OpenAI-compatible API server
│   ├── chat_openai_api.py     # API client example
│   └── web_demo.py            # Streamlit web UI
│
├── trained_models/            # Pre-trained model weights
│   ├── MiniMind2/             # Main model (104M params)
│   ├── MiniMind2-Small/       # Small model (26M params)
│   ├── MiniMind2-MoE/         # Mixture of Experts (145M params)
│   └── MiniMind2-R1/          # Reasoning model
│
├── eval_llm.py                # Model evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # Comprehensive documentation

```

---

## 🤖 Model Architecture

### Core Design (Dense Model)
MiniGPT follows the **Transformer Decoder-Only** architecture similar to Llama 3.1:

**Key Features:**
1. **RMSNorm:** Pre-normalization before each transformer sub-layer
2. **SwiGLU Activation:** Replaces ReLU for better performance
3. **Rotary Position Embedding (RoPE):** No absolute position embeddings
4. **Grouped-Query Attention (GQA):** Reduces memory usage
5. **Flash Attention:** Optional for efficient attention computation

### Model Variants

| Model | Params | Vocab | RoPE θ | Layers | d_model | KV Heads | Q Heads | Experts |
|-------|--------|-------|--------|--------|---------|----------|---------|---------|
| MiniMind2-Small | 26M | 6,400 | 1e6 | 8 | 512 | 2 | 8 | - |
| MiniMind2 | 104M | 6,400 | 1e6 | 16 | 768 | 2 | 8 | - |
| MiniMind2-MoE | 145M | 6,400 | 1e6 | 8 | 640 | 2 | 8 | 1+4 |

### MoE (Mixture of Experts) Architecture
- Based on DeepSeek-V2/V3's MixFFN approach
- Uses shared + routed experts for better specialization
- Implements load balancing with auxiliary loss

---

## 📊 Training Pipeline

### 1. **Tokenization**
- **Custom Tokenizer:** 6,400 vocab size (vs. 64k-150k in commercial models)
- **Trade-off:** Smaller vocab = less efficient encoding but lighter embedding layer
- **Training:** BPE-based tokenizer trained on Chinese corpus

### 2. **Pre-training (Learn Knowledge)**
```bash
python trainer/train_pretrain.py
```
- **Dataset:** `pretrain_hq.jsonl` (~1.6GB high-quality text)
- **Source:** DeepCtrl SFT dataset (Chinese portion)
- **Objective:** Next-token prediction (causal language modeling)
- **Output:** `pretrain_*.pth` weights

### 3. **Supervised Fine-Tuning (Learn Conversation)**
```bash
python trainer/train_full_sft.py
```
- **Datasets:** 
  - `sft_mini_512.jsonl` (1.2GB, recommended for quick training)
  - `sft_512.jsonl` / `sft_1024.jsonl` / `sft_2048.jsonl` (larger variants)
- **Format:** Multi-turn conversations with user/assistant roles
- **Output:** `full_sft_*.pth` weights

### 4. **LoRA Fine-Tuning (Parameter Efficient)**
```bash
python trainer/train_lora.py
```
- **Method:** Low-Rank Adaptation (native PyTorch implementation)
- **Use Cases:** Domain adaptation (medical, self-identity)
- **Datasets:** `lora_medical.jsonl`, `lora_identity.jsonl`

### 5. **RLHF - Direct Preference Optimization**
```bash
python trainer/train_dpo.py
```
- **Dataset:** `dpo.jsonl` (~55MB preference pairs)
- **Source:** Magpie-DPO dataset (Llama3.1-generated)
- **Method:** Offline preference learning (chosen/rejected pairs)
- **Output:** `dpo_*.pth` weights

### 6. **RLAIF - Reinforcement Learning**

#### PPO (Proximal Policy Optimization)
```bash
python trainer/train_ppo.py
```
- **Components:** Actor, Critic, Reference model, Reward model (4 models)
- **Objective:** Maximize reward while staying close to reference policy

#### GRPO (Group Relative Policy Optimization)
```bash
python trainer/train_grpo.py
```
- **Advantage:** Only 2 models needed (policy + reference)
- **Method:** Group-based advantage estimation with KL penalty

#### SPO (Self-Play Optimization)
```bash
python trainer/train_spo.py
```
- **Features:** Adaptive baseline, no grouping, Beta distribution tracking
- **Status:** Experimental (from recent research)

### 7. **Model Distillation**
```bash
python trainer/train_distill_reason.py
```
- **Purpose:** Transfer knowledge from larger reasoning models (e.g., DeepSeek-R1)
- **Dataset:** `r1_mix_1024.jsonl` (distilled reasoning traces)
- **Output:** `reason_*.pth` weights

---

## 📦 Data Pipeline

### Dataset Overview

| Dataset | Size | Purpose | Format |
|---------|------|---------|--------|
| `pretrain_hq.jsonl` | 1.6GB | Pre-training | Plain text |
| `sft_mini_512.jsonl` ✨ | 1.2GB | Quick SFT | Conversations |
| `sft_512.jsonl` | 7.5GB | Full SFT | Conversations |
| `sft_1024.jsonl` | 5.6GB | Medium SFT | Conversations |
| `sft_2048.jsonl` | 9GB | Long-context SFT | Conversations |
| `dpo.jsonl` ✨ | 55MB | Preference learning | Chosen/Rejected pairs |
| `r1_mix_1024.jsonl` | 340MB | Reasoning distillation | CoT conversations |
| `rlaif-mini.jsonl` | 1MB | Online RL | Sampled conversations |
| `lora_medical.jsonl` | 34MB | Domain adaptation | Medical QA |
| `lora_identity.jsonl` | 23KB | Self-recognition | Identity QA |

✨ = Recommended for quick start

### Data Sources
1. **DeepCtrl Dataset:** High-quality Chinese SFT data
2. **Magpie-Align:** Qwen2/2.5 and Llama3.1 generated conversations
3. **R1 Distillation:** DeepSeek-R1 reasoning traces

---

## 🚀 Training Recommendations

### Quick Start (2 hours on RTX 3090)
```bash
# Step 1: Pre-train (learn knowledge)
python trainer/train_pretrain.py

# Step 2: Supervised fine-tune (learn conversation)
python trainer/train_full_sft.py

# Step 3: Test the model
python eval_llm.py --weight full_sft
```

**Datasets Needed:** 
- `pretrain_hq.jsonl` (1.6GB)
- `sft_mini_512.jsonl` (1.2GB)

**Cost:** ~$3 USD (GPU rental) | **Time:** ~2 hours | **Result:** Functional chat model 😊😊

### Full Training (Best Quality)
Use all available datasets (~20GB total)

**Cost:** 💰💰💰💰💰💰💰💰 | **Result:** 😊😊😊😊😊😊

### Middle Ground
Mix `pretrain_hq.jsonl` + `sft_1024.jsonl` or similar combinations

**Cost:** 💰💰💰 | **Result:** 😊😊😊😊

---

## 🎯 Key Features & Innovations

### 1. **Pure PyTorch Implementation**
- All algorithms coded from scratch (no black-box frameworks)
- Educational transparency: understand every line
- Compatible with `transformers`, `trl`, `peft` libraries

### 2. **Multi-GPU Support**
```bash
# DDP (Distributed Data Parallel)
torchrun --nproc_per_node N train_xxx.py

# Optional: WandB/SwanLab logging
python train_xxx.py --use_wandb
```

### 3. **Resume Training**
```bash
python train_xxx.py --from_resume 1
```
- Automatic checkpoint detection
- Supports GPU count changes
- Continuous WandB logging

### 4. **RoPE Extrapolation (YaRN)**
```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```
- Extends context length beyond training limit
- 4x extrapolation factor (2048 → 8192 tokens)

### 5. **Third-Party Ecosystem Support**
- **llama.cpp:** C++ inference with quantization
- **vLLM:** High-throughput serving
- **Ollama:** Local LLM runner (`ollama run jingyaogong/minimind2`)
- **Llama-Factory:** Alternative training framework

### 6. **OpenAI-Compatible API**
```bash
python scripts/serve_openai_api.py
```
Integrates with FastGPT, Open-WebUI, Dify, etc.

---

## 📈 Performance Evaluation

### Benchmark Results (C-Eval, CMMLU, A-CLUE, TMMLU+)

| Model | Params | C-Eval | CMMLU | A-CLUE | TMMLU+ |
|-------|--------|--------|-------|--------|--------|
| MiniMind2 | 104M | 26.52 | 24.42 | 24.97 | 25.27 |
| MiniMind2-Small | 26M | 26.37 | 24.97 | 25.39 | 24.63 |
| MiniMind2-MoE | 145M | 26.6 | 25.01 | 24.83 | 25.01 |

**Note:** Scores are close to random guessing (25%) due to:
- Minimal pre-training data
- No benchmark-specific tuning
- Focus on educational value over leaderboard chasing

### Subjective Quality Ranking
Based on conversational tests, third-party evaluation (DeepSeek-R1 as judge):

1. **MiniMind2 (104M)** - Most complete and accurate responses
2. **MiniMind2-MoE (145M)** - Good balance of specialization
3. **MiniMind2-Small (26M)** - Impressive for size but limited knowledge

---

## 🛠️ Technical Highlights

### Model Configuration
```python
class MiniMindConfig(PretrainedConfig):
    hidden_size: int = 512          # Embedding dimension
    num_hidden_layers: int = 8      # Transformer layers
    num_attention_heads: int = 8    # Query heads
    num_key_value_heads: int = 2    # KV heads (GQA)
    vocab_size: int = 6400          # Tokenizer vocab
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0   # RoPE base frequency
    use_moe: bool = False           # Enable MoE architecture
```

### Training Features
- **Mixed Precision:** FP16/BF16 support
- **Gradient Checkpointing:** Memory optimization
- **DeepSpeed Integration:** (optional)
- **Dynamic Learning Rate:** Cosine decay with warmup
- **Early Stopping:** Based on validation loss

### Distillation Approach
- **Black-box Distillation:** Learn from model outputs (no internal access)
- **Reasoning Traces:** Include `<think>` tags for chain-of-thought
- **Temperature Scaling:** Soften teacher logits for better transfer

---

## 🔬 Research Contributions

### Papers Using MiniMind
1. **ECG-Expert-QA:** Medical LLM evaluation benchmark
2. **Binary-Integer-Programming:** MoE load balancing algorithms
3. **LegalEval-Q:** Legal text quality evaluation
4. **ICML 2025:** Generalization of next-token prediction
5. **FedBRB (TMC 2025):** Federated learning with heterogeneous devices

### Book Reference
- 《从零开始写大模型：从神经网络到Transformer》(Tsinghua University Press)

---

## 📚 Dependencies

**Core:**
- `torch==2.6.0` (PyTorch framework)
- `transformers==4.57.1` (HuggingFace integration)
- `datasets==3.6.0` (Data loading)

**Training:**
- `trl==0.13.0` (Transformer RL)
- `peft==0.7.1` (Parameter-efficient fine-tuning)
- `wandb==0.18.3` or `swanlab==0.6.8` (Experiment tracking)

**Serving:**
- `Flask==3.0.3` (API server)
- `streamlit==1.50.0` (Web UI)
- `openai==1.59.6` (API compatibility)

**Utilities:**
- `jieba==0.42.1` (Chinese tokenization)
- `tiktoken==0.10.0` (Token counting)
- `einops==0.8.1` (Tensor operations)

---

## 🎓 Educational Value

### What Makes This Project Special

1. **Transparent Implementation**
   - All algorithms implemented from scratch
   - No hidden abstractions or magic
   - Comments in Chinese and English

2. **Complete Pipeline**
   - From tokenizer training to RLHF
   - Covers entire LLM development lifecycle
   - Real-world best practices

3. **Resource Efficient**
   - Runnable on consumer hardware
   - <$5 budget for full reproduction
   - Optimized for learning, not production

4. **Comprehensive Documentation**
   - 1900+ line README with detailed explanations
   - Training logs and visualization
   - Troubleshooting guides

5. **Active Community**
   - 11K+ GitHub stars
   - Regular updates and bug fixes
   - Responsive to issues and PRs

---

## 🔄 Recent Updates (2025-10-24)

- ✅ Added RLAIF algorithms: PPO, GRPO, SPO (native implementations)
- ✅ Resume training support with checkpoint recovery
- ✅ YaRN RoPE extrapolation for long contexts
- ✅ Adaptive Thinking for reasoning models
- ✅ Tool calling and reasoning tags in chat template
- ✅ SwanLab integration (WandB alternative for China)
- ✅ Code refactoring and bug fixes

---

## 🎯 Use Cases

### Learning & Education
- Understanding LLM architecture and training
- Experimenting with different training strategies
- Teaching material for NLP courses

### Research
- Baseline for small model studies
- Testing new optimization algorithms
- Benchmarking efficiency techniques

### Domain Adaptation
- Fine-tuning for specific industries (medical, legal, finance)
- Building vertical AI assistants
- Knowledge injection through continued training

### Prototyping
- Quick POC for LLM-based applications
- Testing prompts and conversation flows
- Integration testing with RAG systems

---

## ⚠️ Limitations

1. **Knowledge Base:** Limited pre-training data leads to knowledge gaps
2. **Benchmark Performance:** Not competitive with commercial models
3. **Context Length:** Default 2K tokens (extendable with RoPE scaling)
4. **Multilingual:** Primarily Chinese, limited English capability
5. **Safety:** No built-in content filtering or alignment

**Recommendation:** Use for learning/research, not production applications requiring accuracy.

---

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt
```

### Quick Test (Pre-trained Model)
```bash
# Download model
git clone https://huggingface.co/jingyaogong/MiniMind2

# Run evaluation
python eval_llm.py --load_from ./MiniMind2
```

### Train From Scratch
```bash
# Download datasets
# Place in ./dataset/ directory

# Pre-train
python trainer/train_pretrain.py

# SFT
python trainer/train_full_sft.py

# Evaluate
python eval_llm.py --weight full_sft
```

---

## 🌟 Community & Support

- **GitHub:** https://github.com/jingyaogong/minimind
- **HuggingFace:** https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5
- **ModelScope:** https://www.modelscope.cn/profile/gongjy
- **Issues:** Active issue tracker for bugs and questions
- **Contributions:** PRs welcome for improvements

---

## 📄 License

Apache 2.0 License - Free for commercial and academic use

---

## 🙏 Acknowledgments

The project draws inspiration from:
- Meta's Llama 3.1
- Karpathy's llama2.c
- DeepSeek V2/V3 architecture
- TinyLlama project
- Multiple open-source Chinese LLM initiatives

**Special Recognition:** 11K+ stars, 130+ forks, active contributor community

---

## 📝 Citation

```bibtex
@misc{minimind,
  title={MiniMind: Train a Tiny LLM from scratch},
  author={Jingyao Gong},
  year={2024},
  howpublished={https://github.com/jingyaogong/minimind}
}
```

---

## 🎉 Conclusion

**MiniGPT (MiniMind)** is an exceptional educational resource that demystifies LLM training. It proves that with clever engineering and focused scope, powerful language models can be built with minimal resources. The project's commitment to transparency and accessibility makes it an invaluable tool for students, researchers, and practitioners looking to understand modern AI systems.

**Key Takeaways:**
- ✅ Full LLM pipeline from tokenization to RLHF
- ✅ Affordable training (<$5 budget)
- ✅ Pure PyTorch implementation (no black boxes)
- ✅ Production-ready ecosystem integration
- ✅ Active community and continuous improvements

**Recommendation:** Highly suitable for educational purposes, research experiments, and as a foundation for custom domain-specific language models.

---

**Report Generated:** November 9, 2025  
**Analyzer:** GitHub Copilot CLI  
**Note:** Originally named "minimind", repository was renamed to "miniGPT"
