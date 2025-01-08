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
    text_token = tokenizer.tokenize(' '.join(text))
    assert len(text) == len(rationale), '[!] len(text) != len(rationale) | {} != {}\n{}\n{}'.format(len(text), len(rationale), text, rationale)
    
    rat_token = []
    for t, r in zip(text, rationale):
        token = tokenizer.tokenize(t)
        rat_token += [r]*len(token)

    assert len(text_token) == len(rat_token), "#token != #target rationales of {}".format(id)
    return rat_token

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


def save_checkpoint(args, losses, embedding_layer, trained_model, tokenizer=None):
    # checkpoint = {
    #     'args': args,
    #     'model_state': model_state,
    #     'optimizer_state': optimizer_state
    # }
    file_name = args.exp_name + '.ckpt'
    save_path = os.path.join(args.dir_result, file_name)
    trained_model.save_pretrained(save_directory=save_path)
    if tokenizer:
        tokenizer.save_pretrained(save_directory=save_path)

    args.waiting += 1
    if losses[-1] <= min(losses):
        print("Loss has been decreased from {:.6f} to {:.6f}".format(min(losses[:-1]) if len(losses) > 1 else losses[-1], losses[-1]))
        args.waiting = 0
        best_path = os.path.join(args.dir_result, 'BEST_' + file_name)
        trained_model.save_pretrained(save_directory=best_path)
        if tokenizer:
            tokenizer.save_pretrained(save_directory=best_path)
        
        if args.intermediate == 'mrp':
            # Save the embedding layer params
            emb_file_name = args.exp_name + '_emb.ckpt'
            torch.save(embedding_layer.state_dict(), os.path.join(args.dir_result, emb_file_name))

        print("[!] The best checkpoint is updated")

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