import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import os  

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim

from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification

from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.utils.logging_utils import setup_logging

from huggingface_hub import HfApi

def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')

def add_tokens_to_tokenizer(args, tokenizer):

    special_tokens_dict = {'additional_special_tokens': 
                            ['@USER', '<URL>']}  
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer

# Label Encoding
le = LabelEncoder()
def encode(data):
    return le.fit_transform(data)

def decode(data):
    return le.inverse_transform(data)

def get_token_rationale(tokenizer, text, rationale, id):
    """
    # Example usage
    text = "good movie"              # Input words
    rationale = [1, 0]               # Rationale per word
    # If "good" gets tokenized to ["go", "##od"]
    # Output would be: [1, 1, 0]         # Rationale mapped to each token
    """
    original_text = ' '.join(text)
    text_token = tokenizer.tokenize(original_text)
    assert len(text) == len(rationale), '[!] len(text) != len(rationale) | {} != {}\n{}\n{}'.format(len(text), len(rationale), text, rationale)
    
    rat_token = []
    for t, r in zip(text, rationale):
        token = tokenizer.tokenize(t)
        rat_token += [r]*len(token)

    assert len(text_token) == len(rat_token), "#token != #target rationales of {}".format(id)
    return rat_token , text_token

class GetLossAverage(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()  # type -> int
        v = v.data.sum().item()  # type -> float
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def aver(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def save_checkpoint(args, losses, embedding_layer, trained_model, tokenizer=None, metrics=None):
    intermediate_label = args.intermediate if args.intermediate != False else ''
    per_masked_f1_label = f"_masked_f1_{metrics['val/masked_f1']:.6f}" if 'val/masked_f1' in metrics else ''
    file_name = f"{args.pretrained_model}_{args.finetuning_stage}_{intermediate_label}_val_loss_{metrics['val/loss']:.6f}_ep{metrics['epoch']}_stp{metrics['step']}_f1_{metrics['val/f1']:.6f}_{per_masked_f1_label}.ckpt"
    save_path = os.path.join(args.dir_result, file_name)
    trained_model.save_pretrained(save_directory=save_path)
    if tokenizer:
        tokenizer.save_pretrained(save_directory=save_path)
    
    # Save metrics to a text file in a readable format
    metrics_file = os.path.join(args.dir_result, file_name ,'metrics.txt')
    with open(metrics_file, 'w') as f:
        # Write main metrics
        f.write(f"Pre-Finetuned Model Checkpoint Path: {args.pre_finetuned_model}\n")
        f.write(f"Validation Loss: {metrics['val/loss']:.6f}\n")
        f.write(f"Validation Accuracy: {metrics['val/accuracy']:.6f}\n")
        f.write(f"Validation F1: {metrics['val/f1']:.6f}\n")
        f.write(f"Validation Time: {metrics['val/time']:.2f}s\n")
        f.write("\nClassification Report:\n")
        f.write(str(metrics['val/classification_report']))
        
        # Write MRP specific metrics if present
        if 'val/masked_accuracy' in metrics:
            f.write("\n\nMasked Metrics:\n")
            f.write(f"Masked Accuracy: {metrics['val/masked_accuracy']:.6f}\n")
            f.write(f"Masked F1: {metrics['val/masked_f1']:.6f}\n")
            f.write("\nMasked Classification Report:\n")
            f.write(str(metrics['val/masked_classification_report']))
        
        f.write(f"\n\nEpoch: {metrics['epoch']}\n")
        f.write(f"Step: {metrics['step']}\n")
    
    if args.intermediate == 'mrp':
        # Save the embedding layer params
        emb_file_name = file_name + '_emb_layer_states.ckpt'
        emb_save_path = os.path.join(args.dir_result, emb_file_name)
        torch.save(embedding_layer.state_dict(), emb_save_path)

    args.waiting += 1
    if losses[-1] <= min(losses):
        print(f"[!] Loss has been decreased from {losses[-2] if len(losses) > 1 else losses[-1]:.6f} to {losses[-1]:.6f}")
        args.waiting = 0
    
    # Push to Hugging Face Hub
    huggingface_repo_url = None
    if args.push_to_hub:
        try:
            commit_message = f"""Epoch {metrics['epoch']}, Step {metrics['step']}, Val Loss {metrics['val/loss']:.4f}
            intermediate task: {intermediate_label} F1 {metrics['val/f1']:.4f} {args.wandb_run_url}"""
            repo_name = f"{args.model_name}-{args.finetuning_stage}-{intermediate_label}-{args.exp_date}"
            trained_model.push_to_hub(
                repo_name,
                commit_message= commit_message
            )[2]
            
            # Upload metrics file to hub
            with open(metrics_file, 'r') as f:
                metrics_content = f.read()
            
            # Create model card with metrics
            model_card = f"""
            # Model Details
            - Model: {args.pretrained_model}
            - Finetuning Stage: {args.finetuning_stage}
            - Intermediate Task: {intermediate_label}
            - Wandb Run URL: {args.wandb_run_url}
            
            ## Training Metrics
            ```
            {metrics_content}
            ```
            """
            
            # Push model card to hub
            with open(os.path.join(save_path, "README.md"), 'w') as f:
                f.write(model_card)
            
            huggingface_repo_url = f"https://huggingface.co/s-haturusinghe/{repo_name}"
            # if args.intermediate == 'mrp' then push the embedding layer to heggungface repo
            if args.intermediate == 'mrp':
                api = HfApi()
                api.upload_file(
                path_or_fileobj= emb_save_path,
                path_in_repo= f"{emb_file_name}",
                repo_id= f"s-haturusinghe/{repo_name}",
                repo_type= "model",
            )
                
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {str(e)}")
            print("Continuing without pushing to hub...")
    
    return save_path, huggingface_repo_url


def cleanup_useless_checkpoints(args):
    # Remove checkpoints that are not the best
    checkpoints = os.listdir(args.dir_result)
    best_checkpoint = max([float(x.split('_')[-1].split('.ckpt')[0]) for x in checkpoints if '.ckpt' in x])
    for checkpoint in checkpoints:
        if '.ckpt' in checkpoint:
            if float(checkpoint.split('_')[-1].split('.ckpt')[0]) != best_checkpoint:
                os.remove(os.path.join(args.dir_result, checkpoint))

def load_checkpoint(args, load_best=True, path=None):
    """Load saved checkpoint including model, embedding layer 
    
    Args:
        args: Arguments containing experiment info
        load_best: Whether to load the best checkpoint
        path (str): Path to the saved checkpoint
    
    Returns:
        model: Loaded model
        embedding_layer: Loaded embedding layer (if MRP)
    """

    logger = setup_logging()

    if path is not None:
        # pre_finetune/08012025-0942_LK_xlm-roberta-base_mrp_5e-05_16_600_seed42_pre/08012025-0942_LK_xlm-roberta-base_mrp_5e-05_16_600_seed42_pre.ckpt
        model_path = path + '/' + path.split('/')[-1] + '.ckpt'
        emb_path = path + '/' + path.split('/')[-1] + '_emb.ckpt'

        if load_best:
            model_path = path + '/' + 'BEST_' + path.split('/')[-1] + '.ckpt'


    # Load the model
    model = None
    embedding_layer = None
    logger.info("[MODEL_LOAD] Loading model from {}".format(model_path))
    logger.info("[EMB_LOAD] Loading embedding layer from {}".format(emb_path))
    if args.intermediate == 'rp':
        model = XLMRobertaForTokenClassification.from_pretrained(model_path)
    elif args.intermediate == 'mrp':
        logger.info("Loading model from {}".format(model_path))
        model = XLMRobertaCustomForTCwMRP.from_pretrained(model_path)
        embedding_layer = nn.Embedding(args.n_tk_label, 768)
        embedding_layer.load_state_dict(torch.load(emb_path))

    if model is None:
        raise ValueError("Failed to load model from " + model_path)
        
    return model, embedding_layer



def get_checkpoint_path(args):
    path = args.model_path
    if path is not None:
        # pre_finetune/08012025-0942_LK_xlm-roberta-base_mrp_5e-05_16_600_seed42_pre/08012025-0942_LK_xlm-roberta-base_mrp_5e-05_16_600_seed42_pre.ckpt
        model_path = path + '/' + path.split('/')[-1] + '.ckpt'
        emb_path = path + '/' + path.split('/')[-1] + '_emb.ckpt'
        model_path_best  = path + '/' + 'BEST_' + path.split('/')[-1] + '.ckpt'
        return model_path, emb_path , model_path_best



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def setup_experiment_name(args):
    lm = '-'.join(args.pretrained_model.split('-')[:])
    name = f"{args.exp_date}_{args.lr}_{args.batch_size}_{args.val_int}_seed{args.seed}"
    
    if args.finetuning_stage == 'pre':
        name += f"_{lm}_{args.intermediate}_pre"
    elif args.finetuning_stage == 'final':
        args.intermediate = False
        args.num_labels = int(args.num_labels)
        name += "_final"
        print("Pre-finetuned model path: ", args.pre_finetuned_model)
    
    return name

def setup_directories(args):
    if args.test:
        # Extract experiment name from test model path
        exp_name = args.test_model_path.split('/')[-2]
        base_dir = os.path.join(args.finetuning_stage + "_finetune", exp_name)
        result_dir = os.path.join(base_dir, 'test')
    else:
        exp_name = setup_experiment_name(args)
        result_dir = os.path.join(args.finetuning_stage + "_finetune", exp_name)
    
    os.makedirs(result_dir, exist_ok=True)
    return exp_name, result_dir


