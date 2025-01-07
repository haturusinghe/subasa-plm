import torch
import random


def prepare_gts(args, max_len, bi_rats_str):
    """
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

