from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    output_dir: str = "outputs"
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000

    def __init__(self, model_name, max_length, batch_size, learning_rate, epochs, output_dir):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir
