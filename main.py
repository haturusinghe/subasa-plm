# Standard library imports 
import argparse
import gc
import json
import logging
import os
import random 
import sys
import time
from datetime import datetime
from math import ceil, floor
from pathlib import Path

# Third party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
import wandb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Local imports
from src.dataset.dataset import SOLDDataset
from src.evaluate.evaluate import evaluate, evaluate_for_hatespeech
from src.evaluate.lime import TestLime
from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.utils.helpers import (
    GetLossAverage,
    NumpyEncoder, 
    add_tokens_to_tokenizer,
    get_checkpoint_path,
    get_device,
    load_checkpoint,
    save_checkpoint,
    setup_experiment_name,
    setup_directories
)
from src.utils.logging_utils import setup_logging
from src.utils.prefinetune_utils import add_pads, make_masked_rationale_label, prepare_gts
import subprocess
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def login_to_wandb():
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        raise ValueError("WANDB_API_KEY not set in environment variables")
    try:
        result = subprocess.run(['wandb', 'login', api_key], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to login to wandb: {str(e)}")

def login_to_huggingface():
    api_key = os.getenv('HF_TOKEN')
    if not api_key:
        raise ValueError("HF_TOKEN not set in environment variables")
    try:
        result = subprocess.run(['huggingface-cli', 'login', api_key], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to login to Hugging Face: {str(e)}")


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
    parser.add_argument('--test_model_path', type=str, required=False, help='the checkpoint path to test', default=None)

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
    parser.add_argument('--intermediate', choices=['mrp', 'rp', 'mlm'], default=False, required=False, help='choice of an intermediate task')

    ## Masked Ratioale Prediction 
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--n_tk_label', type=int, default=2)
    
    parser.add_argument('--skip_empty_rat', default=False, help='skip empty rationales', type=bool, required=False)

    # Weights & Biases config
    parser.add_argument('--wandb_project', type=str, default='subasa-llm', help='Weights & Biases project name')
    parser.add_argument('--push_to_hub', default=False, help='save the model to huggingface', type=bool)
    
    #### FOR STEP 2 ####
    parser.add_argument('--pre_finetuned_model', required=False, default=None) # path to the pre-finetuned model
    parser.add_argument('--num_labels', type=int, default=2) # number of classes in the dataset

    ## Explainability based metrics
    parser.add_argument('--explain_sold', default=False, help='Generate Explainablity Metrics', type=bool)
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
        },
        name=args.exp_name
    )

    args.wandb_run_url = wandb.run.get_url()

    # Set seed
    set_seed(args.seed)

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    model = None

    if args.intermediate == 'rp':
        model = XLMRobertaForTokenClassification.from_pretrained(args.pretrained_model)
        emb_layer = None
    elif args.intermediate == 'mlm':
        model = XLMRobertaForMaskedLM.from_pretrained(args.pretrained_model)
        emb_layer = None
    elif args.intermediate == 'mrp':
        model = XLMRobertaCustomForTCwMRP.from_pretrained(args.pretrained_model) 
        emb_layer = nn.Embedding(args.n_tk_label, 768)
        model.config.output_attentions=True
    
    model.resize_token_embeddings(len(tokenizer))

    data_collator = None
    if args.intermediate == 'mlm':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_ratio)

    # Define dataloader
    train_dataset = SOLDDataset(args, 'train', tokenizer=tokenizer) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = SOLDDataset(args, 'val', tokenizer=tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.intermediate == 'mlm':
        train_dataloader.collate_fn = data_collator
        val_dataloader.collate_fn = data_collator

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

            if args.intermediate != 'mlm':
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
                gts = prepare_gts(args, max_len, rationales_batch) # returns ground truth rationale strings with padding added to match the max_len the text (after tokenization) in the batch
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer)
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps
                out_tensor = model(**in_tensor, labels=gts_tensor)
            elif args.intermediate == 'mlm':
                batch = {k: v.to(args.device) for k, v in batch.items()}
                out_tensor = model(**batch)


            loss = out_tensor.loss
            loss.backward()
            optimizer.step()
            get_tr_loss.add(loss)

            # validation model during training
            # TODO : Make sure there is a final validation right after the end of the final epoch
            if i == 0 or (i+1) % args.val_int == 0 or (epoch == args.epochs-1 and (i == steps_per_epoch or i == steps_per_epoch-1)):
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
                if args.intermediate != 'mlm':
                    print("Classification Report:\n", report)

                if args.intermediate == 'mrp':
                    print("* acc about masked: {} | f1 about masked: {}".format(acc[1], f1[1]))
                
                if args.intermediate == 'mrp':
                    print("Classification Report for Masked:\n", report_for_masked)

                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(val_loss, val_time))
                log.write("* acc: {} | f1: {}\n".format(acc[0], f1[0]))
                if args.intermediate == 'mrp':
                    log.write("* acc about masked: {} | f1 about masked: {}\n".format(acc[1], f1[1]))
                log.write("Classification Report:\n{}\n".format(report))
                if args.intermediate == 'mrp':
                    log.write("Classification Report for Masked:\n{}\n".format(report_for_masked))


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
                    
                metrics.update({
                    "epoch": epoch,
                    'step': i,
                })

                if args.intermediate == 'mlm':
                    # remove classificaion metrics from the metrics dict
                    metrics.pop("val/classification_report")
                    metrics.update({
                        "val/classification_report": None,
                    })

                wandb.log(metrics)

                save_path, huggingface_repo_url = save_checkpoint(args, val_losses, emb_layer, model, metrics=metrics)

                #update wandb config with the huggingface repo url and save path of checkpoint
                wandb.config.update({
                    "checkpoint": save_path,
                    "huggingface_repo_url": huggingface_repo_url,
                }, allow_val_change=True)

                

            if args.waiting > args.patience:
                print("early stopping")
                break
        
        if args.waiting > args.patience:
            break
    
    log.close()
    wandb.finish()


def test_mrp(args):

    wandb.init(
        project=args.wandb_project,
        config={
            "batch_size": args.batch_size,
            "intermediate_task": args.intermediate,
            "model": args.pretrained_model,
            "test_model_path": args.test_model_path,
            "seed": args.seed,
            "dataset": args.dataset,
            "finetuning_stage": args.finetuning_stage,
            "val_int": args.val_int,
            "patience": args.patience,
            "mask_ratio": args.mask_ratio,
            "n_tk_label": args.n_tk_label,
            "test": args.test,
            "exp_name": args.exp_name,
            "skip_empty_rat": args.skip_empty_rat,
        },
        name= args.exp_name + '_TEST'
    )
    set_seed(args.seed)

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    if args.intermediate == 'rp':
        model = XLMRobertaForTokenClassification.from_pretrained(args.test_model_path)
        emb_layer = None
    elif args.intermediate == 'mrp':
        model = XLMRobertaCustomForTCwMRP.from_pretrained(args.test_model_path) 
        emb_layer = nn.Embedding(args.n_tk_label, 768)
        loaded_state_dict = torch.load(args.test_model_path + '_emb_layer_states.ckpt')
        emb_layer.load_state_dict(loaded_state_dict)
        model.config.output_attentions=True

    model.resize_token_embeddings(len(tokenizer))
    
    test_dataset = SOLDDataset(args, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    mlb = MultiLabelBinarizer()
    
    if args.intermediate == 'mrp':
        emb_layer.to(args.device)

    model.to(args.device)
    # make directory for test results
    os.makedirs(os.path.join(args.test_model_path, 'test'), exist_ok=True)
    log = open(os.path.join(args.test_model_path, 'test' ,'test_res.txt'), 'a')

    # calculate total number of steps per epoch
    steps_per_epoch = ceil(len(test_dataset) / args.batch_size)
    print("Steps per epoch (testing): ", steps_per_epoch)
    
    losses, loss_avg, time_avg, acc, f1, report, report_for_masked = evaluate(args, model, test_dataloader, tokenizer, emb_layer, mlb)

    print("\nCheckpoint: ", args.test_model_path)
    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}".format(loss_avg, min(losses), max(losses), time_avg))
    print("Acc: {} | F1: {} \n".format(acc[0], f1[0]))

    if args.intermediate == 'mrp':
        print("* acc about masked: {} | f1 about masked: {}".format(acc[1], f1[1]))

    print("Classification Report:\n", report)
    if args.intermediate == 'mrp':
        print("Classification Report for Masked:\n", report_for_masked)
        print('\n')

    log.write("Checkpoint: {} \n".format(args.test_model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {} \n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("Acc: {} | F1: {} \n".format(acc[0], f1[0]))
    log.write("Classification Report:\n{}\n".format(report))
    if args.intermediate == 'mrp':
        log.write("Classification Report for Masked:\n{}\n".format(report_for_masked))
        log.write('\n')
    

    # Log validation metrics
    metrics = {
        "test/Loss_avg": loss_avg,
        "test/Loss_min": min(losses),
        "test/Loss_max": max(losses),
        "test/accuracy": acc[0],
        "test/f1": f1[0],
        "test/time": time_avg,
        "test/classification_report": report,
    }

    if args.intermediate == 'mrp':
        metrics.update({
            "test/masked_accuracy": acc[1],
            "test/masked_f1": f1[1],
            "test/masked_classification_report": report_for_masked,
        })

    # log the path of the checkpoint
    metrics.update({"test/checkpoint": args.test_model_path})

    
    wandb.log(metrics)

    log.close()



def train_offensive_detection(args):
    # Setup logging
    logger = setup_logging()
    logger.info("[START] [FINAL] [OFFENSIVE_LANG_DET] Starting with args: {}".format(args))

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "model": args.pretrained_model,
            "seed": args.seed,
            "dataset": args.dataset,
            "finetuning_stage": args.finetuning_stage,
            "val_int": args.val_int,
            "patience": args.patience,
            "label_classes": args.num_labels,
            "skip_empty_rat": args.skip_empty_rat,
            "pre_finetuned_model": args.pre_finetuned_model,
            "test": args.test,
            "explain_sold": args.explain_sold,
            "mask_ratio_of_pre_finetuned_model": args.mask_ratio,
        },
        name=args.exp_name + '_TRAIN'
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

    steps_per_epoch = ceil(len(train_dataset) / args.batch_size)
    print("Steps per epoch: ", steps_per_epoch)

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
            # TODO : Make sure there is a final validation right after the end of the final epoch
            if i==0 or (i+1) % args.val_int == 0 or (epoch == args.epochs-1 and (i == steps_per_epoch or i == steps_per_epoch-1)):
                _, loss_avg, acc_avg, per_based_scores, time_avg, _ , class_report, all_inputs_and_their_predictions = evaluate_for_hatespeech(args, model, val_dataloader, tokenizer)

                f1_macro, auroc, wandb_roc_curve, roc_curve_values = per_based_scores

                # Unpack the ROC curve values
                fpr, tpr, thresholds = roc_curve_values

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
                # print classification report in terminal
                print("Classification Report:\n", class_report)

                
                log.write("[Epoch {} | Val #{}]\n".format(epoch, args.n_eval))
                log.write("* tr_loss: {}\n".format(tr_loss))
                log.write("* val_loss: {} | val_consumed_time: {}\n".format(loss_avg[0], time_avg))
                log.write("* acc: {} | f1: {} | AUROC: {}\n\n".format(acc_avg[0], per_based_scores[0], per_based_scores[1]))
                log.write("Classification Report:\n{}\n".format(class_report))

                metrics = {
                    "train/loss": tr_loss,
                    "val/loss": loss_avg[0],
                    "val/accuracy": acc_avg[0],
                    "val/f1": per_based_scores[0],
                    "val/auroc": per_based_scores[1],
                    "val/time": time_avg,
                    "val/classification_report": class_report,
                    "epoch": epoch,
                    "step": i,
                }

                # Log metrics to wandb
                wandb.log(metrics)
                # wandb.log({"val/roc" : wandb_roc_curve})

                save_checkpoint(args, val_losses, None, model, metrics=metrics)

            if args.waiting > args.patience:
                print("[!] Early stopping")
                break
        if args.waiting > args.patience:
            break
    log.close()
    wandb.finish()


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


def test_for_hate_speech(args):
    set_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        config={
            "batch_size": args.batch_size,
            "mask_ratio_of_pre_finetuned_model": args.mask_ratio,
            "model": args.pretrained_model,
            "test_model_path": args.test_model_path,
            "seed": args.seed,
            "dataset": args.dataset,
            "finetuning_stage": args.finetuning_stage,
            "val_int": args.val_int,
            "patience": args.patience,
            "top_k": args.top_k,
            "lime_n_sample": args.lime_n_sample,
            "label_classes": args.num_labels,
            "test": args.test,
            "exp_name": args.exp_name,
            "explain_sold": args.explain_sold,
        },
        name=args.exp_name + '_TEST'
    )

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
    model = XLMRobertaForSequenceClassification.from_pretrained(args.test_model_path, num_labels=args.num_labels)
    tokenizer = add_tokens_to_tokenizer(args, tokenizer)

    test_dataset = SOLDDataset(args, 'test')

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.to(args.device)
    model.config.output_attentions=True

    log = open(os.path.join(args.dir_result, 'test_res_performance.txt'), 'a')


    losses, loss_avg, acc, per_based_scores, time_avg, explain_dict_list , class_report, all_inputs_and_their_predictions = evaluate_for_hatespeech(args, model, test_dataloader, tokenizer)

    f1_macro, auroc, wandb_roc_curve, roc_curve_values = per_based_scores

    # Unpack the ROC curve values
    fpr, tpr, thresholds = roc_curve_values

    print("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}\n".format(loss_avg, min(losses), max(losses), time_avg))
    print("** Performance-based Scores **")
    print("Acc: {} | F1: {} | AUROC: {} \n".format(acc[0], f1_macro, auroc))
    # print classification report like a nice table in terminal
    print("Classification Report:\n", class_report)

    log.write("Checkpoint: {}\n".format(args.test_model_path))
    log.write("Loss_avg: {} / min: {} / max: {} | Consumed_time: {}\n\n".format(loss_avg, min(losses), max(losses), time_avg))
    log.write("** Performance-based Scores **\n")
    log.write("Acc: {} | F1: {} | AUROC: {} \n".format(acc[0], f1_macro, auroc))
    log.write("Classification Report:\n{}\n".format(class_report))
    log.close()

    # Log metrics to wandb
    wandb.log({
        "test/Loss_avg": loss_avg,
        "test/Loss_min": min(losses),
        "test/Loss_max": max(losses),
        "test/accuracy": acc[0],
        "test/f1": per_based_scores[0],
        "test/auroc": per_based_scores[1],
        "test/time": time_avg,
        "test/classification_report": class_report,
    })

    wandb.log({"roc" : wandb_roc_curve})

    #save all_inputs_and_their_predictions as a json file local directory
    with open(args.dir_result + '/all_inputs_and_their_predictions.json', 'w') as f:
        all_inputs_and_their_predictions_json = json.dumps(all_inputs_and_their_predictions, cls=NumpyEncoder)
        f.write(all_inputs_and_their_predictions_json)


    # save all_inputs_and_their_predictions as a table in wandb
    # create Table object by iterating over the all_inputs_and_their_predictions
    wandb_table_columns = list(all_inputs_and_their_predictions[0].keys())
    wandb_table_data = [list(i.values()) for i in all_inputs_and_their_predictions]
    wandb_table = wandb.Table(data=wandb_table_data, columns=wandb_table_columns)
    wandb.log({"Table_all_inputs_and_their_predictions": wandb_table})
    


    if args.explain_sold:
        
        with open(args.dir_result + '/for_explain_union.json', 'w') as f:
            f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in explain_dict_list))

        print('[*] Start LIME test')
        lime_tester = TestLime(args)
        lime_dict_list = lime_tester.test(args)  # This could take a little long time
        with open(args.dir_result + '/for_explain_lime.json', 'w') as f:
            f.write('\n'.join(json.dumps(i,cls=NumpyEncoder) for i in lime_dict_list))
        
        # TODO :  Implement the get_explain_results(args) function in src/evaluate/explain_results_lime.py
        # get_explain_results(args)  # The test_res_explain.txt file will be written


if __name__ == '__main__':
    args = parse_args()
    args.device = get_device()
    args.waiting = 0
    args.n_eval = 0

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    # Setup experiment name and paths
    lm = '-'.join(args.pretrained_model.split('-')[:])
    now = datetime.now()
    args.exp_date = (now.strftime('%d%m%Y-%H%M') + '_LK')

    # Setup experiment name and directories
    args.exp_name, args.dir_result = setup_directories(args)
    print("Checkpoint path: ", args.dir_result)

    # login_to_wandb() 
    # login_to_huggingface()

    # Execute appropriate training/testing function based on configuration
    if args.finetuning_stage == 'pre':
        if args.test and args.test_model_path:
            args.batch_size = 1
            test_mrp(args)
        elif not args.test:
            train_mrp(args)
    elif args.finetuning_stage == 'final':
        if args.test and args.test_model_path:
            args.explain_sold = False  # Turn it to True for explainable metrics | WIP
            args.batch_size = 1
            test_for_hate_speech(args)
        elif not args.test:
            train_offensive_detection(args)


