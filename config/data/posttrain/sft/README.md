# SFT (Supervised Fine-Tuning) Configs

This directory contains dataset mixture configurations for supervised fine-tuning.

## Available Configs

### 1. `general.yaml` - General Instruction Following
- **Dataset**: `tatsu-lab/alpaca` (52K instructions)
- **Focus**: Diverse task completion
- **Samples**: 50K
- **Use**: Broad instruction following capability

### 2. `code.yaml` - Code Generation
- **Dataset**: `sahil2801/CodeAlpaca-20k` (20K code examples)
- **Focus**: Programming and code explanation
- **Samples**: 20K
- **Use**: Improve coding abilities

### 3. `reasoning.yaml` - Math & Logic
- **Dataset**: `gsm8k` (7.5K math problems)
- **Focus**: Mathematical reasoning
- **Samples**: 7.5K
- **Use**: Enhance problem-solving skills

## Usage

```bash
# Prepare SFT dataset
python scripts/prepare_dataset.py \
    --config config/data/posttrain/sft/general.yaml \
    --output_dir dataset/

# Train with SFT
python trainer/train_full_sft.py \
    --data_config config/data/posttrain/sft/general.yaml \
    --from_weight out/midtrain_512.pth \
    --use_prepared \
    --epochs 2 \
    --batch_size 16 \
    --device cuda:0
```

## Format

SFT datasets expect `conversations` field:
```json
{
  "conversations": [
    {"role": "user", "content": "instruction..."},
    {"role": "assistant", "content": "response..."}
  ]
}
```

## Dataset Sources

- **Alpaca**: Simple instruction-response pairs
- **CodeAlpaca**: Programming-focused instructions
- **GSM8K**: Grade-school math problems with solutions
