bash ```
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
│   │   ├── config.py                   
│   │   ├── modeling.py                 
│   │   └── modeling_moe.py             
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
│   ├── mixer.py                        
│   ├── loader.py                       # Multi-source loading
│   ├── filters.py                      # Quality filters
│   ├── samplers.py                     # Sampling strategies
│   ├── validation.py                   # Dataset validation
│   └── dataset.md                      # Documentation
│
├── trainer/                             # Training scripts
│   ├── __init__.py
│   ├── train_pretrain.py               
│   ├── train_full_sft.py               
│   ├── train_dpo.py                    
│   ├── train_lora.py
│   ├── train_ppo.py
│   ├── train_grpo.py
│   ├── train_spo.py
│   ├── train_distillation.py
│   └── trainer_utils.py                
│
├── scripts/                             # Utility scripts
│   ├── train.py                        
│   ├── prepare_dataset.py              
│   ├── analyze_mixture.py              
│   ├── evaluate.py                     
│   ├── analyze_training.py             
│   ├── compare_experiments.py         
│   ├── resume_training.py              
│   ├── serve_openai_api.py
│   ├── chat_openai_api.py
│   ├── web_demo.py
│   └── train_tokenizer.py
│
├── evaluation/                          # Evaluation framework
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
├── experiments/                         # Experiment outputs
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
│   ├── TODO.md                         
│   ├── ARCHITECTURE_PLAN.md            
│   ├── DATASET_MIXING_GUIDE.md         
│   ├── MODEL_ARCHITECTURE_GUIDE.md     
│   ├── TRAINING_GUIDE.md               
│   └── SCALING_TO_7B.md                
│
├── notebooks/                           # Tutorial notebooks
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/                              
│   ├── test_mixer.py
│   ├── test_models.py
│   └── test_training.py
│
├── requirements.txt
├── README_en.md
└── LICENSE
```