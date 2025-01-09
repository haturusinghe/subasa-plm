# Subasa - Pretrained Language Models (PLM)

A framework for adapting Pretrained Language Models for Low-Resourced Offensive Language Detection in Sinhala using pretrained models and intermediate tasks.

## Features

- Two-stage finetuning approach with intermediate tasks
- Support for multiple pretrained models (XLM-RoBERTa base/large)
- Intermediate tasks: Masked Rationale Prediction (MRP) and Rationale Prediction (RP)
- Comprehensive evaluation metrics including AUROC and explainability measures
- Integration with Weights & Biases for experiment tracking
- LIME-based model explanations

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
├── main.py              # Main training and evaluation script
├── src/             
│   ├── config/         # Configuration files
│   ├── dataset/        # Dataset loading and processing
│   ├── evaluate/       # Evaluation metrics and explainers
│   ├── models/         # Model implementations
│   └── utils/          # Helper functions and utilities
├── pre_finetune/       # Pre-finetuning stage outputs
└── final_finetune/     # Final stage model outputs
```

## Training Modes

### 1. Pre-finetuning Stage

Train with intermediate tasks (MRP or RP):

```bash
python main.py \
    --pretrained_model xlm-roberta-base \
    --intermediate mrp \
    --val_int 250 \
    --patience 3 \
    --mask_ratio 0.5 \
    --n_tk_label 2 \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.00002 \
    --seed 42 \
    --wandb_project your-wandb-project-name \
    --finetuning_stage pre \
    --dataset sold \
    --skip_empty_rat True

```

### 2. Final Finetuning Stage

Finetune for offensive language detection:

```bash
python main.py \
    --pretrained_model xlm-roberta-base \
    --val_int 250 \
    --patience 3 \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.00002 \
    --seed 42 \
    --wandb_project your-wandb-project-name \
    --finetuning_stage final \
    --dataset sold \
    --num_labels 2 \
    --pre_finetuned_model path/to/prefinetuned/checkpoint
```

## Evaluation

Test a trained model:

```bash
python main.py \
    --test True \
    --model_path path/to/checkpoint \
    --intermediate mrp \
    --batch_size 16
```

## Configuration Options

### General Settings
- `--seed`: Random seed (default: 42)
- `--dataset`: Dataset choice ('sold' or 'hatexplain')
- `--batch_size`: Training batch size
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--patience`: Early stopping patience

### Model Configuration
- `--pretrained_model`: Base model ('xlm-roberta-large' or 'xlm-roberta-base')
- `--finetuning_stage`: Training stage ('pre' or 'final')
- `--intermediate`: Intermediate task ('mrp' or 'rp')
- `--mask_ratio`: Mask ratio for MRP task
- `--num_labels`: Number of classification labels

### Evaluation Settings
- `--top_k`: Top k attention values for explainable metrics
- `--lime_n_sample`: Number of samples for LIME explainer
- `--val_int`: Validation interval

### Logging
- `--wandb_project`: Weights & Biases project name
- `--save_to_hf`: Save model to HuggingFace hub

## Metrics and Evaluation

### Training Metrics
- Loss (training and validation)
- Accuracy
- F1 Scores
- Precision
- Recall
- AUROC (for final stage)

### Explainability Metrics
- LIME-based explanations
- Attention-based analysis
- Classification reports

## Results Storage

Results are saved in experiment-specific directories:
- `pre_finetune/`: Pre-finetuning stage outputs
- `final_finetune/`: Final stage model outputs
- Detailed logs and metrics in `train_res.txt` and `test_res.txt`
- Explainability results in JSON format

## Experiment Tracking

All experiments are logged to Weights & Biases with:
- Training/validation metrics
- Model configurations
- ROC curves and confusion matrices
- System performance metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.

