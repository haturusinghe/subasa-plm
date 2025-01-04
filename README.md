# Subasa - Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala

A modular pipeline for training and evaluating XLM-RoBERTa models using PyTorch and Hugging Face Transformers.

## Project Structure

```
subasa-llm/
├── src/
│   ├── config/
│   │   └── config.py
│   ├── training/
│   │   └── trainer.py
│   ├── data/
│   │   └── dataset.py
│   ├── evaluation/
│   │   └── evaluator.py
│   └── utils/
│       └── logging_utils.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your model parameters in `src/config/config.py`
2. Prepare your dataset using the `TextDataset` class
3. Initialize the trainer and start training
4. Evaluate the model using the `ModelEvaluator`

Example usage will be provided in upcoming examples.