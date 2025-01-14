import torch
import random


def prepare_gts(args, max_len, bi_rats_str):
    """
    returns ground truth rationale strings with padding added to match the max_len the text (after tokenization) in the batch
    args : starting args from main.py
    max_len : max_len of the input_ids tensor
    bi_rats_str : batch of rationale strings
    """
    gts = []
    for bi_rat_str in bi_rats_str:
        bi_list = bi_rat_str.split(',') # since rationale is a string of comma separated values of 0s and 1s we split it to get the list of 0s and 1s
        bi_rat = [int(b) for b in bi_list] # convert the list of strings to list of integers
        
        if args.intermediate == 'rp':
            bi_rat = [0]+bi_rat # add eos to the beginning of the list
            n_pads = max_len - len(bi_rat)  # num of eos + pads
            bi_gt = bi_rat + [0]*n_pads # add pads to the end of the list
        elif args.intermediate == 'mrp':
            bi_gt = [0]+bi_rat+[0] # add eos to the beginning and end of the list , 0 because we are not interested in the first and last default tokens

        gts.append(bi_gt)

    return gts # returns a list of the original rationale strings with padding added to match the max_len of the input_ids tensor


###### MRP ######
def make_masked_rationale_label(args, gt_labels, emb_layer):
    """
    args : starting args from main.py
    labels : batch of rationale strings converted to list with padding added to match the max_len of the input_ids tensor (output of the prepare_gts function)
    emb_layer : the embedding layer of the model

    """
    label_reps_list = []
    masked_idxs_list = []
    masked_labels_list = []
    for label in gt_labels:
        idxs = list(range(len(label)))
        # if args.test:
        if args.intermediate == 'rp': #TODO: check if this is correct, if not remove this block (during testing we probably need to follow same procedure as training)
            masked_idxs = idxs[1:-1]
            masked_label = [-100]+label[1:-1]+[-100]
            label_rep = torch.zeros(len(label), emb_layer.embedding_dim)
        else:  # Validation and Training
            masked_idxs = random.sample(idxs[1:-1], int(len(idxs[1:-1])*args.mask_ratio)) # according to the mask ratio we randomly select a number of indices in the rationale list to mask
            masked_idxs.sort()
            label_tensor = torch.tensor(label).to(args.device) # convert the rationale list to a tensor
            label_rep = emb_layer(label_tensor) # pass the tensor to the embedding layer to get the embeddings of the rationale tokens
            label_rep[0] = torch.zeros(label_rep[0].shape) # set the first token to zero since we are not interested in them
            label_rep[-1] = torch.zeros(label_rep[-1].shape) # set the last token to zero since we are not interested in them
            for i in masked_idxs:
                label_rep[i] = torch.zeros(label_rep[i].shape) # go through the list of indices to mask and set the embeddings of the tokens at those indices to zero (mask them)
            
            # For loss
            masked_label = []
            for j in idxs:
                if j in masked_idxs:
                    masked_label.append(label[j])
                else:
                    masked_label.append(-100)
            
        assert len(masked_label) == label_rep.shape[0], '[!] len(masked_label) != label_rep.shape[0] | \n{} \n{}'.format(masked_label, label_rep)
        
        masked_idxs_list.append(masked_idxs) # list of indices of the rationale tokens that were masked
        masked_labels_list.append(masked_label) # labels of the rationale tokens that were masked (ground truth)
        label_reps_list.append(label_rep) # list of tensors of embeddings of the rationale tokens with the masked tokens set to zero vectors

    return masked_idxs_list, label_reps_list, masked_labels_list
    

def add_pads(args, max_len, labels, masked_labels, label_reps):
    assert len(labels) == len(masked_labels) == len(label_reps), '[!] add_pads | different total nums {} {} {}'.format(len(labels), len(masked_labels), len(label_reps))
    labels_pad, masked_labels_pad, label_reps_pad = [], [], []
    for label_for_each_token, label_with_masks, token_embeddings_with_masks in zip(labels, masked_labels, label_reps):
        assert len(label_for_each_token) == len(label_with_masks) == token_embeddings_with_masks.shape[0], '[!] add_pads | different lens of each ele {} {} {}'.format(len(label_for_each_token), len(label_with_masks), token_embeddings_with_masks.shape[0])
        if args.test:
            labels_pad.append(label_for_each_token)
            masked_labels_pad.append(label_with_masks)
            label_reps_pad.append(token_embeddings_with_masks)
        else:
            n_pads = max_len - len(label_for_each_token) # get the length of the maximmum length tokonized sequence
            label_for_each_token = label_for_each_token + [0]*n_pads # add padds to make each sequence the same length
            label_with_masks = label_with_masks + [-100]*n_pads # add padds to make each sequence the same length , -100 is the default ignore_index in PyTorch’s CrossEntropyLoss. So, any token with a label of -100 will be ignored in loss computation
            zero_ten = torch.zeros(n_pads, args.hidden_size).to(args.device)
            token_embeddings_with_masks = torch.cat((token_embeddings_with_masks, zero_ten), 0)
            
            assert len(label_for_each_token) == len(label_with_masks) == token_embeddings_with_masks.shape[0], '[!] add_pads | different lens of each ele'
            labels_pad.append(label_for_each_token)
            masked_labels_pad.append(label_with_masks)
            label_reps_pad.append(token_embeddings_with_masks)

    return labels_pad, masked_labels_pad, label_reps_pad
