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
from src.utils.logging_utils import setup_logging

class SOLDDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.train_dataset_path = 'SOLD_DATASET/sold_train_split.json' 
        self.test_dataset_path = 'SOLD_DATASET/sold_test_split.json'

        self.label_list = ['NOT' , 'OFF']
        self.label_count = [0, 0]
        self.logger = setup_logging()

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

        if args.check_errors == True:
            # this flag is to make the datasets very small to quickly run the training to see errors
            self.dataset = self.dataset[:100]

        if args.intermediate and args.skip_empty_rat:
            rm_idxs = []
            removed_items = []
            for idx, d in enumerate(self.dataset):
                if 1 not in json.loads(d['rationales']) and d['label'] == "OFF":
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                removed_items.append(self.dataset[j])
                del self.dataset[j]
            self.logger.info(f"[DATASET] [MODE: {mode}] [SKIP_EMPTY_RAT] Removed {len(removed_items)} items")
            self.logger.info(f"[DATASET] [MODE: {mode}] [REMOVED_ITEMS] {str(removed_items)}")
        
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

            length_of_rationales = len(rationales)
            length_of_text = len(text.split())
            
            # # convert ratianles back to a string and make sure its same as the original raw_rationale_from_ds
            # back_to_str = "[" + ", ".join([str(r) for r in rationales]) + "]"
            # assert raw_rationale_from_ds == back_to_str, "Rationales are not the same after conversion"

            # there are items in the dataset with label is NOt and the rationale list is an empty arry. so we need to make sure that the length of the rationales is the same as the length of the text
            if label == "NOT" and len(rationales) == 0:
                # create a list of zeros with the same length as the text
                rationales = [0] * length_of_text

            length_of_rationales = len(rationales)
            if length_of_rationales != length_of_text:
                self.logger.error(f"[ERROR] [RAT_LEN] {length_of_rationales} [TEXT_LEN] {length_of_text} [ID] {id}")
                sys.exit(1)

            text_str_split_to_tokens = text.split(' ')
            final_rationale_tokens, text_after_tokenizer = get_token_rationale(self.tokenizer, copy.deepcopy(text_str_split_to_tokens), copy.deepcopy(rationales), copy.deepcopy(id))

            tmp = []
            for r in final_rationale_tokens:
                tmp.append(str(r))
            final_rationales_str = ','.join(tmp)
            return (text, cls_num, final_rationales_str)

        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)
        
        else:
            return ()