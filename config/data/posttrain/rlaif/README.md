# RLAIF (Reinforcement Learning from AI Feedback) Configs

This directory contains prompt datasets for online RL methods.

## Available Configs

### 1. `ppo.yaml` - PPO Training
- **Dataset**: `tatsu-lab/alpaca` (instructions as prompts)
- **Focus**: Diverse prompts
- **Samples**: 30K prompts
- **Use**: PPO with reward model

### 2. `grpo.yaml` - Group Relative PO
- **Dataset**: `tatsu-lab/alpaca`
- **Focus**: Group preferences
- **Samples**: 30K prompts
- **Use**: GRPO training

### 3. `spo.yaml` - Self-Play Optimization
- **Dataset**: `tatsu-lab/alpaca`
- **Focus**: Self-improvement
- **Samples**: 25K prompts
- **Use**: SPO training

## Usage

```bash
# Prepare RLAIF prompts
python scripts/prepare_dataset.py \
    --config config/data/posttrain/rlaif/ppo.yaml \
    --output_dir dataset/

# Train with PPO
python trainer/train_ppo.py \
    --data_config config/data/posttrain/rlaif/ppo.yaml \
    --from_weight out/full_sft_512.pth \
    --use_prepared \
    --epochs 1 \
    --batch_size 16 \
    --device cuda:0
```

## Format

RLAIF datasets use prompts (just the user query):
```json
{
  "conversations": [
    {"role": "user", "content": "prompt text"}
  ]
}
```

The model generates responses online during training.

## Dataset Source

- **Alpaca**: Using instruction field as prompts
- Models generate multiple responses
- Reward model or self-evaluation ranks them
- Policy optimized via RL
