import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import time
import random

from sklearn.metrics import classification_report, f1_score, accuracy_score,roc_auc_score, roc_curve

from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage, save_checkpoint
from src.utils.prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads

import wandb

def get_pred_cls(logits):
    probs = F.softmax(logits, dim=1)
    #labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    max_probs = np.max(probs, axis=1).tolist()
    probs = probs.tolist()
    pred_clses = []
    for m, p in zip(max_probs, probs):
        pred_clses.append(p.index(m))
    
    return probs, pred_clses

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
    
    time_avg = consumed_time / len(dataloader)
    loss_avg = [sum(losses) / len(dataloader)]
    acc = [accuracy_score(total_gt_clses, total_pred_clses)]
    f1 = f1_score(total_gt_clses, total_pred_clses, average='macro')
    class_report = classification_report(total_gt_clses, total_pred_clses, output_dict=True)

    # Use probabilities of positive class (class 1)
    positive_class_probs = total_probs[:, 1]
    auroc = roc_auc_score(total_gt_clses, positive_class_probs)
    roc_curve_values = roc_curve(total_gt_clses, positive_class_probs)

    wandb_roc_curve = wandb.plot.roc_curve( total_gt_clses, total_pred_clses,
                        labels=['NOT','OFF'])

    per_based_scores = [f1, auroc, wandb_roc_curve ,roc_curve_values]

    return losses, loss_avg, acc, per_based_scores, time_avg, explain_dict_list, class_report



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