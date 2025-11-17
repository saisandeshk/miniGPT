# DPO (Direct Preference Optimization) Configs

This directory contains dataset configs for preference-based alignment.

## Available Configs

### 1. `helpfulness.yaml` - Helpfulness Preferences
- **Dataset**: `Anthropic/hh-rlhf` (helpful subset)
- **Focus**: Quality and helpfulness
- **Samples**: 50K preference pairs
- **Use**: Align for better responses

### 2. `safety.yaml` - Safety Preferences
- **Dataset**: `Anthropic/hh-rlhf` (harmless subset)
- **Focus**: Safe, harmless outputs
- **Samples**: 50K preference pairs
- **Use**: Reduce harmful content

## Usage

```bash
# Prepare DPO dataset
python scripts/prepare_dataset.py \
    --config config/data/posttrain/dpo/helpfulness.yaml \
    --output_dir dataset/

# Train with DPO
python trainer/train_dpo.py \
    --data_config config/data/posttrain/dpo/helpfulness.yaml \
    --from_weight out/full_sft_512.pth \
    --use_prepared \
    --epochs 1 \
    --batch_size 16 \
    --device cuda:0
```

## Format

DPO datasets need `chosen` and `rejected` fields:
```json
{
  "chosen": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "good response"}
  ],
  "rejected": [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "bad response"}
  ]
}
```

## Dataset Source

- **HH-RLHF**: Human preference data from Anthropic
  - Helpful: chosen responses are more helpful
  - Harmless: chosen responses are safer
