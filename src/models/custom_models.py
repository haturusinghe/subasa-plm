import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from torch.nn import CrossEntropyLoss 

from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput

import numpy as np


class XLMRobertaCustomForTCwMRP(XLMRobertaForTokenClassification):
    def __init__(self, config):
        super(XLMRobertaCustomForTCwMRP, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.mrp_classifier = nn.Linear(config.hidden_size, 2)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        mrp_logits = self.mrp_classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            mrp_loss = loss_fct(mrp_logits.view(-1, 2), labels.view(-1))
            loss = loss + mrp_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
