# Subasa-LLM

Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala

## Installation

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

## Usage

The training process consists of two stages:

### 1. Pre-finetuning Stage

For pre-finetuning with Masked Rationale Prediction (MRP):

```bash
python main.py \
    --intermediate mrp \
    --mask_ratio 0.5 \
    --n_tk_label 2 \
    --pretrained_model xlm-roberta-base \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.00005 \
    --val_int 945 \
    --patience 3 \
    --seed 42 \
    --dataset sold \
    --wandb_project your-project-name \
    --finetuning_stage pre
```

For Rationale Prediction (RP):
```bash
python main.py \
    --intermediate rp \
    # ...other parameters same as above...
```

### 2. Final Finetuning Stage

```bash
python main.py \
    --intermediate mrp \
    --pre_finetuned_model path/to/pre-finetuned/model \
    --label_classess 2 \
    --finetuning_stage final \
    # ...other parameters same as pre-finetuning...
```

## Parameters

- `--intermediate`: Choice of intermediate task (`mrp` or `rp`)
- `--mask_ratio`: Ratio of tokens to mask (default: 0.5)
- `--n_tk_label`: Number of token labels (default: 2)
- `--pretrained_model`: Base model (`xlm-roberta-base` or `xlm-roberta-large`)
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: 0.00005)
- `--val_int`: Validation interval (default: 945)
- `--patience`: Early stopping patience (default: 3)
- `--seed`: Random seed (default: 42)
- `--dataset`: Dataset choice (`sold` or `hatexplain`)
- `--wandb_project`: Weights & Biases project name
- `--finetuning_stage`: Training stage (`pre` or `final`)
- `--label_classess`: Number of classes in dataset (default: 2)