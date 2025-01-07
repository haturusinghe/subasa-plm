import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification

import argparse
import logging
from pathlib import Path
import gc
import numpy as np
import os
import sys
from tqdm import tqdm
import time
from datetime import datetime
import random
from sklearn.preprocessing import MultiLabelBinarizer

from src.config.config import ModelConfig
from src.utils.logging_utils import setup_logging
from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage

from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.dataset.dataset import SOLDDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Subasa - Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala')

    #SEED 
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # DATASET
    # TODO : Give option to use dataset from local folder or download from huggingface
    dataset_choices = ['sold', 'hatexplain']
    parser.add_argument('--dataset', default='sold', choices=dataset_choices, help='a dataset to use')
    
    # PRETRAINED MODEL
    model_choices = ['xlm-roberta-large', 'xlm-roberta-base' ]
    parser.add_argument('--pretrained_model', default='xlm-roberta-base', choices=model_choices, help='a pre-trained model to use')  

    # TRAIN
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--val_int', type=int, default=945)  
    parser.add_argument('--patience', type=int, default=3)

    ## Pre-Finetuing Task
    parser.add_argument('--intermediate', choices=['mrp', 'rp'], required=True, help='choice of an intermediate task')

    ## Masked Ratioale Prediction 
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--n_tk_label', type=int, default=2)

    return parser.parse_args()

def train(args):
    # Setup logging
    logger = setup_logging()
    logger.info("Starting with args: {}".format(args))

    # Set seed
    set_seed(args.seed)

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    if args.intermediate == 'rp':
        model = XLMRobertaForTokenClassification.from_pretrained(args.pretrained_model)
        emb_layer = None
    elif args.intermediate == 'mrp':
        model = XLMRobertaCustomForTCwMRP.from_pretrained(args.pretrained_model) 
        emb_layer = nn.Embedding(args.n_tk_label, 768)
        model.config.output_attentions=True
    
    model.resize_token_embeddings(len(tokenizer))

    # Define dataloader
    train_dataset = SOLDDataset(args, 'train') 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = SOLDDataset(args, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    get_tr_loss = GetLossAverage()
    mlb = MultiLabelBinarizer()

    

if __name__ == '__main__':
    args = parse_args()
    args.test = False
    args.check_errors = False
    args.device = get_device()

    lm = '-'.join(args.pretrained_model.split('-')[:])

    now = datetime.now()
    args.exp_date = (now.strftime('%d%m%Y-%H%M') + '_LK')
    args.exp_name = args.exp_date + '_'+ lm + '_' + args.intermediate +  "_"  + str(args.lr) + "_" + str(args.batch_size) + "_" + str(args.val_int)
    
    dir_result = os.path.join("pre_finetune", args.exp_name)
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)

    print("Checkpoint path: ", dir_result)
    args.dir_result = dir_result
    args.waiting = 0
    args.n_eval = 0

    gc.collect()
    torch.cuda.empty_cache()

    train(args)

