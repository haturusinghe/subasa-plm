from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from ast import literal_eval

from transformers import XLMRobertaTokenizer
import copy
import os
import json
import numpy as np
import emoji
import sys
import argparse

from src.utils.helpers import add_tokens_to_tokenizer, get_token_rationale

class SOLDDataset(Dataset):
    # TODO 
    def __init__(self, args, mode='train'):
        self.train_dataset_path = 'SOLD_DATASET/sold_train_split.json' #TODO : Make path dynamic ?
        self.test_dataset_path = 'SOLD_DATASET/sold_test_split.json'

        self.label_list = ['NOT' , 'OFF']
        self.label_count = [0, 0]

        if mode == 'train':
            with open(self.train_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])
            
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['label'] == self.label_list[i]:
                        self.label_count[i] += 1
            
            # TODO : If its train mode, split the train into train and validation from the initial train set

        elif mode == 'test':
            with open(self.test_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])

        if args.intermediate:
            rm_idxs = []
            for idx, d in enumerate(self.dataset):
                if 1 not in json.loads(d['rationales']) and d['label'] == "OFF":
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                del self.dataset[j]
        
        self.mode = mode
        self.intermediate = args.intermediate

        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        self.tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx]['post_id']
        text = self.dataset[idx]['tokens'] #use tokens key instead of text because the length of rationales is the same as tokens
        label = self.dataset[idx]['label']
        cls_num = self.label_list.index(label)
        
        if self.intermediate:
            rationales = json.loads(self.dataset[idx]['rationales'])
            final_rationale_tokens = get_token_rationale(self.tokenizer, copy.deepcopy(text), copy.deepcopy(rationales), copy.deepcopy(id))

            tmp = []
            for r in final_rationale_tokens:
                tmp.append(str(r))
            final_rationales_str = ','.join(tmp)
            return (text, cls_num, final_rationales_str)

        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)
        
        else:
            return ()