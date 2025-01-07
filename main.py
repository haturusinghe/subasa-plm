from math import ceil, floor
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

from sklearn.metrics import f1_score, accuracy_score

from src.config.config import ModelConfig
from src.utils.logging_utils import setup_logging
from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage, save_checkpoint
from src.utils.prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads

from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.dataset.dataset import SOLDDataset

import wandb

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

    parser.add_argument('--check_errors', default=False, help='check errors in the dataset', type=bool)

    # Weights & Biases config
    parser.add_argument('--wandb_project', type=str, default='subasa-llm', help='Weights & Biases project name')


    return parser.parse_args()

def train(args):
    # Setup logging
    logger = setup_logging()
    logger.info("Starting with args: {}".format(args))

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "model": args.pretrained_model,
            "intermediate_task": args.intermediate,
            "n_tk_label": args.n_tk_label,
            "mask_ratio": args.mask_ratio,
            "seed": args.seed,
        },
        name=args.exp_name
    )

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

    if args.intermediate == 'mrp':
        optimizer = optim.RAdam(list(emb_layer.parameters())+list(model.parameters()), lr=args.lr, betas=(0.9, 0.99))
        emb_layer.to(args.device)
        emb_layer.train()
    else:
        optimizer = optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    
    model.to(args.device)
    model.train()

     # configuration = model.config
    log = open(os.path.join(args.dir_result, 'train_res.txt'), 'a')
    #write starting args to log
    log.write(str(args) + '\n')

    tr_losses = []
    val_losses = []
    val_cls_accs = []

    # calculate total number of steps per epoch
    steps_per_epoch = ceil(len(train_dataset) / args.batch_size)
    print("Steps per epoch: ", steps_per_epoch)

    # calculate validation interval based on steps per epoch
    dyanamic_val_int = floor(steps_per_epoch / 3) - 3

    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="TRAINING MODEL for {} | Epoch: {}".format(args.intermediate,epoch), mininterval=0.01)):
            # each row in batch before processing is ordered as follows: (text, cls_num, final_rationales_str) : text is the tweet , cls_num is the label (0 for NOT and 1 for OFF), final_rationales_str is the rationale corresponding to the tokenized text
            in_tensor = tokenizer(batch[0], return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            max_len = in_tensor['input_ids'].shape[1] 

            optimizer.zero_grad()

            if args.intermediate == 'rp':  
                gts = prepare_gts(args, max_len, batch[2])
                gts_tensor = torch.tensor(gts).long().to(args.device)
                out_tensor = model(**in_tensor, labels=gts_tensor)
                
            elif args.intermediate == 'mrp':
                gts = prepare_gts(args, max_len, batch[2])
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer)
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps
                out_tensor = model(**in_tensor, labels=gts_tensor)

            loss = out_tensor.loss
            loss.backward()
            optimizer.step()
            get_tr_loss.add(loss)

            # Log training metrics
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
            })

            # validation model during training
            if i == 0 or (i+1) % dyanamic_val_int == 0:
                _, val_loss, val_time, acc, f1 = evaluate(args, model, val_dataloader, tokenizer, emb_layer, mlb)

                args.n_eval += 1
                model.train()

                val_losses.append(val_loss)
                tr_loss = get_tr_loss.aver()
                tr_losses.append(tr_loss) 
                get_tr_loss.reset()

                print("[Epoch {} | Val #{}]".format(epoch, args.n_eval))
                print("* tr_loss: {}".format(tr_loss))
                print("* val_loss: {} | val_consumed_time: {}".format(val_loss, val_time))
                print("* acc: {} | f1: {}".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    print("* acc about masked: {} | f1 about masked: {}".format(acc[1], f1[1]))
                print('\n')

                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(val_loss, val_time))
                log.write("* acc: {} | f1: {}\n".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    log.write("* acc about masked: {} | f1 about masked: {}\n".format(acc[1], f1[1]))
                log.write('\n')

                # Log validation metrics
                metrics = {
                    "val/loss": val_loss,
                    "val/accuracy": acc[0],
                    "val/f1": f1[0],
                    "val/time": val_time,
                }
                
                if args.intermediate == 'mrp':
                    metrics.update({
                        "val/masked_accuracy": acc[1],
                        "val/masked_f1": f1[1]
                    })
                
                wandb.log(metrics)

                save_checkpoint(args, val_losses, emb_layer, model)

            if args.waiting > args.patience:
                print("early stopping")
                break
        
        if args.waiting > args.patience:
            break
    
    wandb.finish()
    log.close()


          
def evaluate(args, model, dataloader, tokenizer, emb_layer, mlb):
    all_pred_clses, all_pred_clses_masked, all_gts, all_gts_masked_only = [], [], [], []
    losses = []
    consumed_time = 0

    model.eval()
    if args.intermediate == 'mrp':
        emb_layer.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            in_tensor = tokenizer(batch[0], return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            max_len = in_tensor['input_ids'].shape[1]

            if args.intermediate == 'rp':
                gts = prepare_gts(args, max_len, batch[2])
                gts_tensor = torch.tensor(gts).to(args.device)

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor)
                consumed_time += time.time() - start_time   

            elif args.intermediate == 'mrp':
                gts = prepare_gts(args, max_len, batch[2])
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer)
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor)
                consumed_time += time.time() - start_time

            loss = out_tensor.loss.item()
            logits = out_tensor.logits
            pred_probs = F.softmax(logits, dim=2)

            losses.append(loss)
            
            if args.intermediate == 'rp':
                pred_probs = pred_probs.detach().cpu().numpy()
                pred_clses = np.argmax(pred_probs, axis=2)
                pred_clses = pred_clses.tolist()
                all_pred_clses += pred_clses
                all_gts += gts
                
            elif args.intermediate == 'mrp':
                pred_probs = F.softmax(logits, dim=2)
                pred_clses_pad, pred_clses_wo_pad, pred_clses_masked, gts_masked_only = [], [], [], []
                for pred_prob, idxs, gt in zip(pred_probs, masked_idxs, gts):
                    pred_cls = [p.index(max(p)) for p in pred_prob.tolist()]
                    pred_clses_pad += pred_cls

                    if len(pred_cls) == len(gt):
                        pred_cls_wo_pad = pred_cls
                    else:
                        pred_cls_wo_pad = pred_cls[(len(pred_cls)-len(gt)):]
                    pred_clses_wo_pad += pred_cls_wo_pad

                    pred_cls_masked = [pred_cls[i] for i in idxs]
                    gt_masked_only = [gt[i] for i in idxs]
                    pred_clses_masked += pred_cls_masked
                    gts_masked_only += gt_masked_only
                    all_gts += gt    
                
                all_pred_clses += pred_clses_wo_pad
                all_pred_clses_masked += pred_clses_masked
                all_gts_masked_only += gts_masked_only

    loss_avg = sum(losses) / len(dataloader)
    time_avg = consumed_time / len(dataloader)

    if args.intermediate == 'rp':
        all_gts = mlb.fit_transform(all_gts)
        all_pred_clses = mlb.fit_transform(all_pred_clses)
    acc = [accuracy_score(all_gts, all_pred_clses)]
    f1 = [f1_score(all_gts, all_pred_clses, average='macro')]
    if args.intermediate == 'mrp':
        acc.append(accuracy_score(all_gts_masked_only, all_pred_clses_masked))
        f1.append(f1_score(all_gts_masked_only, all_pred_clses_masked, average='macro'))

    return losses, loss_avg, time_avg, acc, f1           

    

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

