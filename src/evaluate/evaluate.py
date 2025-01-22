import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import numpy as np
from tqdm import tqdm
import time
import random

from sklearn.metrics import classification_report, f1_score, accuracy_score,roc_auc_score, roc_curve

from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage, save_checkpoint
from src.utils.prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads

import wandb

def get_pred_cls(logits):
    probs = F.softmax(logits, dim=1) #dim=1 because the logits are of shape (batch_size, num_labels)
    #labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    max_probs = np.max(probs, axis=1).tolist() # we use this to get only the probabilities of the predicted classes, not all classes
    probs = probs.tolist() 
    pred_clses = []
    for m, p in zip(max_probs, probs):
        pred_clses.append(p.index(m)) # index of the maximum probability in the list of probabilities , which is actually the predicted class 0 (NOT) or 1(OFF)
    
    return probs, pred_clses

def evaluate(args, model, dataloader, tokenizer, emb_layer, mlb):
    all_pred_clses, all_pred_clses_masked, all_gts, all_gts_masked_only = [], [], [], []
    masked_predictions, masked_labels = [], []
    losses = []
    consumed_time = 0

    model.eval()
    if args.intermediate == 'mrp':
        emb_layer.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            if args.intermediate != 'mlm':
                input_texts_batch, class_labels_of_texts_batch, rationales_batch = batch[0], batch[1], batch[2]

                in_tensor = tokenizer(input_texts_batch, return_tensors='pt', padding=True)
                in_tensor = in_tensor.to(args.device)
                max_len = in_tensor['input_ids'].shape[1]

            if args.intermediate == 'rp':
                gts = prepare_gts(args, max_len, rationales_batch)
                gts_tensor = torch.tensor(gts).to(args.device)

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor)
                consumed_time += time.time() - start_time   

            elif args.intermediate == 'mrp':
                gts = prepare_gts(args, max_len, rationales_batch)
                masked_idxs, label_reps, masked_gts = make_masked_rationale_label(args, gts, emb_layer)
                gts_pad, masked_gts_pad, label_reps = add_pads(args, max_len, gts, masked_gts, label_reps)

                label_reps = torch.stack(label_reps).to(args.device)
                gts_tensor = torch.tensor(masked_gts_pad).to(args.device)
                in_tensor['label_reps'] = label_reps

                start_time = time.time()
                out_tensor = model(**in_tensor, labels=gts_tensor) #-100 values in the gts_tensor are a flag to ignore them during loss calculation
                consumed_time += time.time() - start_time
            elif args.intermediate == 'mlm':
                batch = {k: v if torch.is_tensor(v) else torch.tensor(v) for k, v in batch.items()}
                batch = {k: v.to(args.device) for k, v in batch.items()}
                start_time = time.time()
                out_tensor = model(**batch)
                consumed_time += time.time() - start_time


            loss = out_tensor.loss.item()
            logits = out_tensor.logits
            pred_probs = F.softmax(logits, dim=2) 
            # pred_probs is a tensor of shape (batch_size, max_len, num_labels) where num_labels is the number of classes in the dataset
            # pred_probs contains the probabilities of each token in the input sequence belonging to each class in the dataset. example : [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]] where 0.1 is the probability of the first token in the input sequence belonging to class 0 and 0.9 is the probability of the first token in the input sequence belonging to class 1
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
            
            elif args.intermediate == 'mlm':
                # For MLM, we need to track the predictions and ground truth only for masked tokens

                pred_probs = F.softmax(logits, dim=2)
                predictions = torch.argmax(logits, dim=2)  # Shape: (batch_size, sequence_length)
                
                # Get the labels (ground truth) from the batch
                labels = batch['labels']  # Shape: (batch_size, sequence_length)
                
                # Process each sequence in the batch
                for pred, label in zip(predictions, labels):
                    # Find positions where tokens were masked (label != -100)
                    masked_positions = label != -100
                    
                    # Get predictions and labels only for masked positions
                    sequence_preds = pred[masked_positions].cpu().tolist()
                    sequence_labels = label[masked_positions].cpu().tolist()
                    
                    masked_predictions.extend(sequence_preds)
                    masked_labels.extend(sequence_labels)

    loss_avg = sum(losses) / len(dataloader)
    time_avg = consumed_time / len(dataloader)

    if args.intermediate == 'rp':
        gts_flat = [item for sublist in all_gts for item in sublist]
        preds_flat = [item for sublist in all_pred_clses for item in sublist]
        all_gts = gts_flat
        all_pred_clses = preds_flat
    #     all_gts = mlb.fit_transform(all_gts)
    #     all_pred_clses = mlb.fit_transform(all_pred_clses)
    elif args.intermediate == 'mlm':
        all_gts = masked_labels
        all_pred_clses = masked_predictions


    acc = [accuracy_score(all_gts, all_pred_clses)] #all_gts and all_pred_clses are lists all ground truth and predicted labels respectively concatanated across all batches into a single list
    f1 = [f1_score(all_gts, all_pred_clses, average='macro')]
    report = classification_report(all_gts, all_pred_clses, output_dict=True)
    
    report_for_masked = None
    if args.intermediate == 'mrp':
        acc.append(accuracy_score(all_gts_masked_only, all_pred_clses_masked))
        f1.append(f1_score(all_gts_masked_only, all_pred_clses_masked, average='macro'))
        report_for_masked = classification_report(all_gts_masked_only, all_pred_clses_masked, output_dict=True)

    return losses, loss_avg, time_avg, acc, f1, report, report_for_masked      


def evaluate_for_hatespeech(args, model, dataloader, tokenizer):
    losses = []
    consumed_time = 0
    total_pred_clses, total_gt_clses, total_probs = [], [], []
    all_inputs_and_their_predictions = []

    explain_dict_list = []
    label_dict = {0:'NOT', 1:'OFF'}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL (Phase 2 for OffensiveDetection) | # {}".format(args.n_eval), mininterval=0.01)):
            input_texts_batch, class_labels_of_texts_batch, ids_batch = batch[0], batch[1], batch[2]

            in_tensor = tokenizer(input_texts_batch, return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            gts_tensor = class_labels_of_texts_batch.to(args.device)

            start_time = time.time()
            out_tensor = model(**in_tensor, labels=gts_tensor)
            consumed_time += time.time() - start_time

            loss = out_tensor.loss
            logits = out_tensor.logits
            attns = out_tensor.attentions[11] 

            losses.append(loss.item())

            probs, pred_clses = get_pred_cls(logits)
            labels_list = class_labels_of_texts_batch.tolist()

            total_gt_clses += labels_list
            total_pred_clses += pred_clses
            total_probs += probs

            # TODO : Complete this
            if args.test and args.explain_sold:
                if labels_list[0] == 0:  # if label is 'NOT'
                    continue                     
                explain_dict = get_dict_for_explain(args, model, tokenizer, in_tensor, gts_tensor, attns, ids_batch[0], label_dict[pred_clses[0]], probs[0])
                if explain_dict == None:
                    continue
                explain_dict_list.append(explain_dict)

            if args.test : 
                # save the input and its prediction for later analysis
                for input_text, pred_cls, gt_cls, prob in zip(input_texts_batch, pred_clses, labels_list, probs):
                    all_inputs_and_their_predictions.append({'input_text': input_text, 'pred_cls': pred_cls, 'gt_cls': gt_cls, 'prob': prob})
    
    time_avg = consumed_time / len(dataloader)
    loss_avg = [sum(losses) / len(dataloader)]
    acc = [accuracy_score(total_gt_clses, total_pred_clses)]
    f1 = f1_score(total_gt_clses, total_pred_clses, average='macro')
    class_report = classification_report(total_gt_clses, total_pred_clses, output_dict=True)

    # Use probabilities of positive class (class 1)
    total_probs = np.array(total_probs)
    positive_class_probs = total_probs[:, 1]
    auroc = roc_auc_score(total_gt_clses, positive_class_probs)
    roc_curve_values = roc_curve(total_gt_clses, positive_class_probs)

    wandb_roc_curve = wandb.plot.roc_curve( total_gt_clses, total_probs,
                        labels=['NOT','OFF'])

    per_based_scores = [f1, auroc, wandb_roc_curve ,roc_curve_values]
    return losses, loss_avg, acc, per_based_scores, time_avg, explain_dict_list, class_report, all_inputs_and_their_predictions



def get_dict_for_explain(args, model, tokenizer, in_tensor, gts_tensor, attns, id, pred_cls, pred_prob):
    explain_dict = {}
    explain_dict["annotation_id"] = id
    explain_dict["classification"] = pred_cls 
    explain_dict["classification_scores"] = {"NOT": pred_prob[0], "OFF": pred_prob[1]}
    
    attns = np.mean(attns[:,:,0,:].detach().cpu().numpy(),axis=1).tolist()[0]
    top_indices = sorted(range(len(attns)), key=lambda i: attns[i])[-args.top_k:]  # including start/end token ?
    temp_hard_rationale = []
    for ind in top_indices:
        temp_hard_rationale.append({'end_token':ind+1, 'start_token':ind})

    gt = gts_tensor.detach().cpu().tolist()[0]
    
    explain_dict["rationales"] = [{"docid": id, 
                                "hard_rationale_predictions": temp_hard_rationale, 
                                "soft_rationale_predictions": attns,
                                #"soft_sentence_predictions": [1.0],
                                #"truth": gts_tensor.detach().cpu().tolist()[0]}, 
                                "truth": gt, 
                                }]

    in_ids = in_tensor['input_ids'].detach().cpu().tolist()[0]
    
    in_ids_suf, in_ids_com = [], []
    for i in range(len(attns)):
        if i in top_indices:
            in_ids_suf.append(in_ids[i])
        else:
            in_ids_com.append(in_ids[i])

    suf_tokens = tokenizer.convert_ids_to_tokens(in_ids_suf)
    suf_text = tokenizer.convert_tokens_to_string(suf_tokens)
    suf_text = suf_text.lower()
    in_ids_suf = tokenizer.encode(suf_text)

    in_ids_com = [101]+in_ids_com[1:-1]+[102]
  
    in_ids_suf = torch.tensor(in_ids_suf)
    in_ids_suf = torch.unsqueeze(in_ids_suf, 0).to(args.device)
    in_ids_com = torch.tensor(in_ids_com)
    in_ids_com = torch.unsqueeze(in_ids_com, 0).to(args.device)
    
    in_tensor_suf = {'input_ids': in_ids_suf, 
                    'token_type_ids': torch.zeros(in_ids_suf.shape, dtype=torch.int).to(args.device), 
                    'attention_mask': torch.ones(in_ids_suf.shape, dtype=torch.int).to(args.device)}
    in_tensor_com = {'input_ids': in_ids_com, 
                    'token_type_ids': torch.zeros(in_ids_com.shape, dtype=torch.int).to(args.device), 
                    'attention_mask': torch.ones(in_ids_com.shape, dtype=torch.int).to(args.device)}
    
    out_tensor_suf = model(**in_tensor_suf, labels=gts_tensor)  
    prob_suf = F.softmax(out_tensor_suf.logits, dim=1).detach().cpu().tolist()[0]
    try:
        out_tensor_com = model(**in_tensor_com, labels=gts_tensor) 
    except:
        print(id)

    prob_com = F.softmax(out_tensor_com.logits, dim=1).detach().cpu().tolist()[0]
    
    explain_dict['sufficiency_classification_scores'] = {"NOT": prob_suf[0], "OFF": prob_suf[1]}
    explain_dict['comprehensiveness_classification_scores'] = {"NOT": prob_com[0], "OFF": prob_com[1]}
    
    return explain_dict