import random 

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
)

# Local imports
from src.dataset.dataset import SOLDDataset
from src.evaluate.evaluate import evaluate, evaluate_for_hatespeech
from src.models.custom_models import XLMRobertaCustomForTCwMRP
from src.utils.helpers import (
    GetLossAverage,
    NumpyEncoder, 
    add_tokens_to_tokenizer,
    get_checkpoint_path,
    get_device,
    save_checkpoint,
)
from src.utils.logging_utils import setup_logging
from src.utils.prefinetune_utils import add_pads, make_masked_rationale_label, prepare_gts

from lime.lime_text import LimeTextExplainer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class TestLime():
    def __init__(self, args):
        self.args = args
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_model)
        model = XLMRobertaForSequenceClassification.from_pretrained(args.model_path, num_labels=args.num_labels)
        tokenizer = add_tokens_to_tokenizer(args, tokenizer)
        
        self.tokenizer = tokenizer
        model.to(args.device)
        model.eval()
        self.model = model

         # Define dataloader
        test_dataset = SOLDDataset(args, 'test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        self.dataloader = test_dataloader

        self.explainer = LimeTextExplainer(class_names=['NOT', 'OFF'], split_expression='\s+', random_state=333, bow=False)
        self.label_dict = {0:'NOT', 1:'OFF'}

    def get_prob(self, texts):  # input -> list
        probs_list = []
        with torch.no_grad():
            for text in texts:
                in_tensor = self.tokenizer(text, return_tensors='pt', padding=True)
                in_tensor = in_tensor.to(self.args.device)
                
                out_tensor = self.model(**in_tensor)  # [batch_size * sequence_length, num_labels]
                logits = out_tensor.logits

                probs = F.softmax(logits, dim=1)
                probs = probs.squeeze(0)
                probs = probs.detach().cpu().numpy()
                probs_list.append(probs)

        return np.array(probs_list)

    def test(self, args):
        lime_dict_list = []
        for i, batch in enumerate(tqdm(self.dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            texts, labels, ids = batch[0], batch[1], batch[2]
            label = labels[0]
            if label == 1:
                continue 
            
            exp = self.explainer.explain_instance(texts[0], self.get_prob, num_features=6, top_labels=3, num_samples=args.lime_n_sample)
            
            temp = {}
            pred_id = np.argmax(exp.predict_proba)
            pred_cls = self.label_dict[pred_id]
            gt_cls = label
            temp["annotation_id"] = ids[0]
            temp["classification"] = pred_cls
            temp["classification_scores"] = {"NOT": exp.predict_proba[0], "OFF": exp.predict_proba[1]}

            attention = [0]*len(texts[0].split(" "))
            exp_res = exp.as_map()[pred_id]
            for e in exp_res:
                if e[1] > 0:
                    attention[e[0]]=e[1]

            final_explanation = [0]
            tokens = texts[0].split(" ")
            for i in range(len(tokens)):
                temp_tokens = self.tokenizer.encode(tokens[i],add_special_tokens=False)
                for j in range(len(temp_tokens)):
                     final_explanation.append(attention[i])
            final_explanation.append(0)
            attention = final_explanation

            #assert(len(attention) == len(row['Attention']))
            topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-args.top_k:]

            temp_hard_rationales=[]
            for ind in topk_indices:
                temp_hard_rationales.append({'end_token':ind+1, 'start_token':ind})

            temp["rationales"] = [{"docid": ids[0], 
                                "hard_rationale_predictions": temp_hard_rationales, 
                                "soft_rationale_predictions": attention,
                                #"soft_sentence_predictions":[1.0],
                                "truth": 0}]

            in_ids = self.tokenizer.encode(texts[0])
    
            in_ids_suf, in_ids_com = [], []
            for i in range(len(attention)):
                if i in topk_indices:
                    in_ids_suf.append(in_ids[i])
                else:
                    in_ids_com.append(in_ids[i])
            
            suf_tokens = self.tokenizer.convert_ids_to_tokens(in_ids_suf)
            suf_text = self.tokenizer.convert_tokens_to_string(suf_tokens)
            suf_text = suf_text.lower()
            
            com_tokens = self.tokenizer.convert_ids_to_tokens(in_ids_com[1:-1])
            com_text = self.tokenizer.convert_tokens_to_string(com_tokens)

            suf_probs = self.get_prob([suf_text])
            com_probs = self.get_prob([com_text])

            temp["sufficiency_classification_scores"] = {"NOT": suf_probs[0][0], "OFF": suf_probs[0][1]}
            temp["comprehensiveness_classification_scores"] = {"NOT": com_probs[0][0], "OFF": com_probs[0][1]}

            lime_dict_list.append(temp)

        return lime_dict_list