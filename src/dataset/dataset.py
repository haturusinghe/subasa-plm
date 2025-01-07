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
    def __init__(self, args, mode='train'):
        self.train_dataset_path = 'SOLD_DATASET/sold_train_split.json' #TODO : Make path dynamic ?
        self.test_dataset_path = 'SOLD_DATASET/sold_test_split.json'

        self.label_list = ['NOT' , 'OFF']
        self.label_count = [0, 0]

        if mode == 'test':
            with open(self.test_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])
        elif mode == 'train' or mode == 'val':
            with open(self.train_dataset_path, 'r') as f:
                self.dataset = list(json.load(f))
            # Sort dataset by a unique identifier to ensure consistent ordering
            self.dataset.sort(key=lambda x: x['post_id'])

            #use train_test_split to split the train set into train and validation
            train_set, val_set = train_test_split(self.dataset, test_size=0.1, random_state=args.seed)

            if mode == 'train':
                self.dataset = train_set
            elif mode == 'val':
                self.dataset = val_set

            
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['label'] == self.label_list[i]:
                        self.label_count[i] += 1

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
            raw_rationale_from_ds = self.dataset[idx]['rationales'] #this is as a string (of a list) in the dataset
            rationales = literal_eval(raw_rationale_from_ds) # converts the raw string to a list of integers
            

            # convert ratianles back to a string and make sure its same as the original raw_rationale_from_ds
            back_to_str = "[" + ", ".join([str(r) for r in rationales]) + "]"
            assert raw_rationale_from_ds == back_to_str, "Rationales are not the same after conversion"

            
            if len(rationales) != len(text.split()):
                rationales = [0] * len(text.split())
            

            final_rationale_tokens = get_token_rationale(self.tokenizer, copy.deepcopy(text.split(' ')), copy.deepcopy(rationales), copy.deepcopy(id))

            tmp = []
            for r in final_rationale_tokens:
                tmp.append(str(r))
            final_rationales_str = ','.join(tmp)
            return (text, cls_num, final_rationales_str)

        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)
        
        else:
            return ()