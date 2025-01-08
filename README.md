# SUBASA - Offensive Language Detection in Sinhala

This project focuses on adapting Language Models for Low-Resourced Offensive Language Detection in Sinhala using pretrained models and intermediate tasks.

## Setup

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
subasa-llm/
├── main.py           # Main training and evaluation script
├── src/             
│   ├── config/      # Configuration files
│   ├── dataset/     # Dataset loading and processing
│   ├── evaluate/    # Evaluation metrics and functions
│   ├── models/      # Custom model implementations
│   └── utils/       # Helper functions and utilities
```

## Training

The project supports two finetuning stages:
1. Pre-finetuning with intermediate tasks (MRP or RP)
2. Final finetuning for offensive language detection

### Pre-finetuning Stage

```bash
python main.py \
    --finetuning_stage pre \
    --intermediate mrp \
    --pretrained_model xlm-roberta-base \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.00005 \
    --val_int 945 \
    --patience 3 \
    --mask_ratio 0.5 \
    --n_tk_label 2
```

### Final Finetuning Stage

```bash
python main.py \
    --finetuning_stage final \
    --pre_finetuned_model /path/to/pretrained/model \
    --pretrained_model xlm-roberta-base \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.00005 \
    --val_int 945 \
    --patience 3 \
    --label_classess 2
```

## Testing

To evaluate a trained model:

```bash
python main.py \
    --test True \
    --model_path /path/to/checkpoint \
    --intermediate mrp \
    --batch_size 16
```

## Configuration Options

- `--seed`: Random seed (default: 42)
- `--dataset`: Dataset choice ('sold' or 'hatexplain')
- `--finetuning_stage`: Training stage ('pre' or 'final')
- `--pretrained_model`: Base model ('xlm-roberta-large' or 'xlm-roberta-base')
- `--intermediate`: Intermediate task ('mrp' or 'rp')
- `--mask_ratio`: Mask ratio for MRP task (default: 0.5)
- `--wandb_project`: Weights & Biases project name
- `--top_k`: Top k attention values for explainable metrics
- `--lime_n_sample`: Number of samples for LIME explainer

## Metrics and Logging

- Training metrics are logged to Weights & Biases
- Results are saved in experiment-specific directories
- Evaluation includes:
  - Accuracy
  - F1 Score
  - Classification Reports
  - AUROC (for final stage)

