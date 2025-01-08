import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import time
import random

from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.utils.helpers import get_device, add_tokens_to_tokenizer, GetLossAverage, save_checkpoint
from src.utils.prefinetune_utils import prepare_gts, make_masked_rationale_label, add_pads

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