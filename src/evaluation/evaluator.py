from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class ModelEvaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, test_dataset):
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        return {
            'classification_report': classification_report(labels, preds),
            'confusion_matrix': confusion_matrix(labels, preds)
        }
