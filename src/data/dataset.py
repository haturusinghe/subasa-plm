from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer
from typing import List, Dict, Any

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], model_name: str, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }
