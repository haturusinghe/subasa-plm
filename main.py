from math import ceil, floor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification , XLMRobertaForTokenClassification,XLMRobertaForMaskedLM

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
from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage, save_checkpoint, load_checkpoint, get_checkpoint_path
from src.utils.prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads
from src.evaluate.evaluate import evaluate, evaluate_for_hatespeech

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
    dataset_choices = ['sold', 'hatexplain']
    parser.add_argument('--dataset', default='sold', choices=dataset_choices, help='a dataset to use')

    # EXPERIMENT FINETUNING STAGE
    finetuning_stage_choices = ['pre', 'final']
    parser.add_argument('--finetuning_stage', default='pre', choices=finetuning_stage_choices, help='a finetuning stage to use')

    # TESTING 
    parser.add_argument('--test', default=False, help='test the model', type=bool)
    parser.add_argument('--model_path', type=str, required=False, help='the checkpoint path to test', default=None)

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
    parser.add_argument('--skip_empty_rat', default=False, help='skip empty rationales', type=bool, required=False)

    # Weights & Biases config
    parser.add_argument('--wandb_project', type=str, default='subasa-llm', help='Weights & Biases project name')
    parser.add_argument('--save_to_hf', default=False, help='save the model to huggingface', type=bool)
    
    #### FOR STEP 2 ####
    parser.add_argument('--pre_finetuned_model', required=False, default=None) # path to the pre-finetuned model
    parser.add_argument('--label_classess', type=int, default=2) # number of classes in the dataset

    ## Explainability based metrics
    parser.add_argument('--top_k', default=5, help='the top num of attention values to evaluate on explainable metrics')
    parser.add_argument('--lime_n_sample', default=100, help='the num of samples for lime explainer')


    return parser.parse_args()

def train_mrp(args):
    # Setup logging
    logger = setup_logging()
    logger.info("[START] Starting with args: {}".format(args))

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
            "dataset": args.dataset,
            "finetuning_stage": args.finetuning_stage,
            "val_int": args.val_int,
            "patience": args.patience,
            "skip_empty_rat": args.skip_empty_rat,
            "check_errors": args.check_errors,
            "top_k": args.top_k,
            "lime_n_sample": args.lime_n_sample,
            "label_classes": args.label_classess,
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
    
    # Log to wandb details about the optimizer
    wandb.config.update({
        "optimizer": optimizer.__class__.__name__,
        "betas": optimizer.defaults['betas'],
        "eps": optimizer.defaults['eps'],
        "weight_decay": optimizer.defaults['weight_decay'],
    })

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

    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="TRAINING MODEL for {} | Epoch: {}".format(args.intermediate,epoch), mininterval=0.01)):
            # each row in batch before processing is ordered as follows: (text, cls_num, final_rationales_str) : text is the tweet , cls_num is the label (0 for NOT and 1 for OFF), final_rationales_str is the rationale corresponding to the tokenized text

            input_texts_batch, class_labels_of_texts_batch, rationales_batch = batch[0], batch[1], batch[2]

            in_tensor = tokenizer(input_texts_batch, return_tensors='pt', padding=True)
            max_len = in_tensor['input_ids'].shape[1] 
            in_tensor = in_tensor.to(args.device)

            optimizer.zero_grad()

            if args.intermediate == 'rp':  
                gts = prepare_gts(args, max_len, rationales_batch)
                gts_tensor = torch.tensor(gts).long().to(args.device)
                out_tensor = model(**in_tensor, labels=gts_tensor)
                
            elif args.intermediate == 'mrp':
                gts = prepare_gts(args, max_len, rationales_batch)
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

            # # Log training metrics
            # wandb.log({
            #     "train/loss": loss.item(),
            #     "train/learning_rate": optimizer.param_groups[0]['lr'],
            #     "epoch": epoch,
            # })

            # validation model during training
            if i == 0 or (i+1) % args.val_int == 0:
                _, val_loss, val_time, acc, f1, report, report_for_masked  = evaluate(args, model, val_dataloader, tokenizer, emb_layer, mlb) # report and report_for_masked are classification reports from sklearn

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
                print("Classification Report:\n", report)
                if args.intermediate == 'mrp':
                    print("Classification Report for Masked:\n", report_for_masked)
                print('\n')

                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(val_loss, val_time))
                log.write("* acc: {} | f1: {}\n".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    log.write("* acc about masked: {} | f1 about masked: {}\n".format(acc[1], f1[1]))
                log.write("Classification Report:\n{}\n".format(report))
                if args.intermediate == 'mrp':
                    log.write("Classification Report for Masked:\n{}\n".format(report_for_masked))
                log.write('\n')

                # Log validation metrics
                metrics = {
                    "val/loss": val_loss,
                    "val/accuracy": acc[0],
                    "val/f1": f1[0],
                    "val/time": val_time,
                    "val/classification_report": report,
                }
                
                if args.intermediate == 'mrp':
                    metrics.update({
                        "val/masked_accuracy": acc[1],
                        "val/masked_f1": f1[1],
                        "val/masked_classification_report": report_for_masked,
                    })
                
                wandb.log(metrics)

                save_checkpoint(args, val_losses, emb_layer, model)

            if args.waiting > args.patience:
                print("early stopping")
                break
        
        if args.waiting > args.patience:
            break
    
    log.close()
    test_trained_model(args, model, tokenizer, emb_layer, mlb)
    wandb.finish()


def test_trained_model(args, model, tokenizer, emb_layer, mlb):
    print("Testing the model after training!!")
    logger = setup_logging()
    logger.info("[START] Testing the model after training!!")
    test_dataset = SOLDDataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    losses, loss_avg, time_avg, acc, f1, report, report_for_masked = evaluate(args, model, test_dataloader, tokenizer, emb_layer, mlb)

    print("\nCheckpoint: ", args.model_path)
    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}".format(loss_avg, min(losses), max(losses), time_avg))
    print("Acc: {} | F1: {} \n".format(acc[0], f1[0]))
    print("Classification Report:\n", report)

    log = open(os.path.join(args.dir_result, 'test_res.txt'), 'a')
    log.write("Checkpoint: {} \n".format(args.model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {} \n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("Acc: {} | F1: {} \n".format(acc[0], f1[0]))

    # Log validation metrics
    metrics = {
        "test/Loss_avg": loss_avg,
        "test/Loss_min": min(losses),
        "test/Loss_max": max(losses),
        "test/accuracy": acc[0],
        "test/f1": f1[0],
        "test/time": time_avg,
    }
    # log the path of the checkpoint
    metrics.update({"test/checkpoint": args.model_path})

    wandb.log(metrics)

    log.close()

def test_mrp(args):

    set_seed(args.seed)
    model_path, emb_path , model_path_best = get_checkpoint_path(args)

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    if args.intermediate == 'rp':
        model = XLMRobertaForTokenClassification.from_pretrained(model_path_best)
        emb_layer = None
    elif args.intermediate == 'mrp':
        model = XLMRobertaCustomForTCwMRP.from_pretrained(model_path_best) 
        emb_layer = nn.Embedding(args.n_tk_label, 768)
        # Load the state dictionary
        loaded_state_dict = torch.load(emb_path)

        # Apply to an embedding layer
        emb_layer.load_state_dict(loaded_state_dict)

        model.config.output_attentions=True

    model.resize_token_embeddings(len(tokenizer))
    
    test_dataset = SOLDDataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    mlb = MultiLabelBinarizer()
    
    if args.intermediate == 'mrp':
        emb_layer.to(args.device)

    model.to(args.device)

    log = open(os.path.join(args.dir_result, 'test_res.txt'), 'a')

    # calculate total number of steps per epoch
    steps_per_epoch = ceil(len(test_dataset) / args.batch_size)
    print("Steps per epoch (testing): ", steps_per_epoch)
    
    losses, loss_avg, time_avg, acc, f1, report, report_for_masked = evaluate(args, model, test_dataloader, tokenizer, emb_layer, mlb)

    print("\nCheckpoint: ", args.model_path)
    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}".format(loss_avg, min(losses), max(losses), time_avg))
    print("Acc: {} | F1: {} \n".format(acc[0], f1[0]))
    print("Classification Report:\n", report)

    log.write("Checkpoint: {} \n".format(args.model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {} \n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("Acc: {} | F1: {} \n".format(acc[0], f1[0]))

    # Log validation metrics
    metrics = {
        "test/Loss_avg": loss_avg,
        "test/Loss_min": min(losses),
        "test/Loss_max": max(losses),
        "test/accuracy": acc[0],
        "test/f1": f1[0],
        "test/time": time_avg,
    }
    # log the path of the checkpoint
    metrics.update({"test/checkpoint": args.model_path})

    
    wandb.log(metrics)

    log.close()



def train_offensive_detection(args):
    # Setup logging
    logger = setup_logging()
    logger.info("[START] [FINAL TRAIN HS] Starting with args: {}".format(args))

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
            "dataset": args.dataset,
            "finetuning_stage": args.finetuning_stage,
            "val_int": args.val_int,
            "patience": args.patience,
            "skip_empty_rat": args.skip_empty_rat,
            "check_errors": args.check_errors,
            "top_k": args.top_k,
            "lime_n_sample": args.lime_n_sample,
            "label_classes": args.label_classess,
        },
        name=args.exp_name
    )

    # Set seed
    set_seed(args.seed)

    model, tokenizer = load_model_train(args)
    model.resize_token_embeddings(len(tokenizer))

    # Define dataloader
    train_dataset = SOLDDataset(args, 'train') 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = SOLDDataset(args, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    get_tr_loss = GetLossAverage()
    optimizer = optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    model.config.output_attentions=True
    model.to(args.device)
    model.train()

    log = open(os.path.join(args.dir_result, 'train_res.txt'), 'a')

    tr_losses, val_losses, val_f1s, val_accs = [], [], [], []
    for epoch in range(args.epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="TRAINING (Phase 2 for OffensiveDetection) | Epoch: {}".format(epoch), mininterval=0.01)):  # data: (post_words, target_rat, post_id)
            input_texts_batch, class_labels_of_texts_batch, ids_batch = batch[0], batch[1], batch[2]

            in_tensor = tokenizer(input_texts_batch, return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            gts_tensor = class_labels_of_texts_batch.to(args.device)

            optimizer.zero_grad()

            out_tensor = model(**in_tensor, labels=gts_tensor)  
            loss = out_tensor.loss
            
            loss.backward()
            optimizer.step()
            get_tr_loss.add(loss)

            # Validation 
            if i==0 or (i+1) % args.val_int == 0:
                _, loss_avg, acc_avg, per_based_scores, time_avg, _ , class_report = evaluate_for_hatespeech(args, model, val_dataloader, tokenizer)

                args.n_eval += 1
                model.train()

                val_losses.append(loss_avg[0])
                val_accs.append(acc_avg[0])
                val_f1s.append(per_based_scores[0])

                tr_loss = get_tr_loss.aver()
                tr_losses.append(tr_loss) 
                get_tr_loss.reset()

                print("[Epoch {} | Val #{}]".format(epoch, args.n_eval))
                print("* tr_loss: {}".format(tr_loss))
                print("* val_loss: {} | val_consumed_time: {}".format(loss_avg[0], time_avg))
                print("* acc: {} | f1: {} | AUROC: {}\n".format(acc_avg[0], per_based_scores[0], per_based_scores[1]))
                
                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(loss_avg[0], time_avg))
                log.write("* acc: {} | f1: {} | AUROC: {}\n\n".format(acc_avg[0], per_based_scores[0], per_based_scores[1]))

                # Log metrics to wandb
                wandb.log({
                    "train/loss": tr_loss,
                    "val/loss": loss_avg[0],
                    "val/accuracy": acc_avg[0],
                    "val/f1": per_based_scores[0],
                    "val/auroc": per_based_scores[1],
                    "val/time": time_avg,
                    "val/classification_report": class_report,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

                save_checkpoint(args, val_losses, None, model)

            if args.waiting > args.patience:
                print("[!] Early stopping")
                break
        if args.waiting > args.patience:
            break

    log.close() 


def load_model_train(args):
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)
    model = XLMRobertaForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=args.num_labels)

    if 'mlm' in args.pre_finetuned_model:
        pre_finetuned_model = XLMRobertaForMaskedLM.from_pretrained(args.pre_finetuned_model) 
    else:
        pre_finetuned_model = XLMRobertaForTokenClassification.from_pretrained(args.pre_finetuned_model)

    model_state = model.state_dict()
    finetuned_state = pre_finetuned_model.state_dict()
    
    # Initialize condition layer randomly 
    filtered_pretrained_state = {}
    for (k1, v1), (k2, v2) in zip(model_state.items(), finetuned_state.items()):
        if v1.size() == v2.size():
            filtered_pretrained_state[k1] = v2
        else:
            filtered_pretrained_state[k1] = v1

    model_state.update(filtered_pretrained_state)
    model.load_state_dict(model_state, strict=True)

    return model, tokenizer


if __name__ == '__main__':
    args = parse_args()
    args.device = get_device()

    lm = '-'.join(args.pretrained_model.split('-')[:])

    now = datetime.now()
    args.exp_date = (now.strftime('%d%m%Y-%H%M') + '_LK')

    if args.test == False:
        args.exp_name = f"{args.exp_date}_{lm}_{args.intermediate}_{args.lr}_{args.batch_size}_{args.val_int}_seed{args.seed}"
        if args.finetuning_stage == 'pre':
            args.exp_name += "_pre"
        elif args.finetuning_stage == 'final':
            args.intermediate = False
            print("Pre-finetuned model path: ", args.pre_finetuned_model)
            args.exp_name += f"_ncls{args.label_classess}_final"
            args.num_labels = int(args.num_labels)


        dir_result = os.path.join(args.finetuning_stage + "_finetune", args.exp_name)
        os.makedirs(dir_result, exist_ok=True)

        print("Checkpoint path: ", dir_result)
        args.dir_result = dir_result
    elif args.test == True:
        args.exp_name = args.model_path.split('/')[-1]
        args.dir_result = os.path.join( args.exp_name, 'test')
        os.makedirs(args.dir_result, exist_ok=True)

    args.waiting = 0
    args.n_eval = 0

    gc.collect()
    torch.cuda.empty_cache()

    if args.finetuning_stage == 'pre' and args.test == False:
        train_mrp(args)
    elif args.finetuning_stage == 'pre' and args.test == True:
        if args.model_path:
            test_mrp(args)
    elif args.finetuning_stage == 'final' and args.test == False:
        train_offensive_detection(args)


