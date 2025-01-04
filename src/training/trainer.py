from transformers import TrainingArguments, Trainer, XLMRobertaForSequenceClassification
from ..config.config import ModelConfig
import torch

class XLMRobertaTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.trainer = None

    def setup_model(self, num_labels: int):
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels
        )

    def train(self, train_dataset, eval_dataset=None):
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset else "no"
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        self.trainer.train()
