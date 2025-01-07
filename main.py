import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

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


from src.config.config import ModelConfig
from src.utils.logging_utils import setup_logging
from src.utils.helpers import get_device

def parse_args():
    parser = argparse.ArgumentParser(description='Subasa - Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala')

    # DATASET
    # TODO : Give option to use dataset from local folder or download from huggingface
    
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

# def main():
#     # Setup logging
#     logger = setup_logging()
#     logger.info("Starting training pipeline")

#     # Parse arguments
#     args = parse_args()

#     # Create configuration
#     config = ModelConfig(
#         model_name=args.model_name,
#         max_length=args.max_length,
#         batch_size=args.batch_size,
#         learning_rate=args.learning_rate,
#         epochs=args.epochs,
#         output_dir=args.output_dir
#     )



#     try:
#         # Train the model
#         logger.info("Starting training")


#         # Evaluate the model
#         logger.info("Starting evaluation")

#         # Log results
#         logger.info("Classification Report:")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}")
#         raise

if __name__ == '__main__':
    args = parse_args()
    args.test = False
    args.device = get_device()

    lm = '-'.join(args.pretrained_model.split('-')[:-1])

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

    print(args)

    # train(args)
