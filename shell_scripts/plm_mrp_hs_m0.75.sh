#!/bin/bash

# Constants for common parameters
BASE_MODEL="xlm-roberta-base"
wandb_p="sold_mrp_to_hs_v2"
MASK_RATIO="0.75"
SEED="13"
COMMON_EXP_NAME="xlmr-sold_mrp${MASK_RATIO}_s${SEED}"

###########################################
## Stage 1: MRP (Masked Region Prediction)
###########################################

# MRP Training
python main.py \
    --pretrained_model "${BASE_MODEL}" --wandb_p "${wandb_p}" \
    --exp_save_name "${COMMON_EXP_NAME}_pre" \
    --finetuning_stage pre \
    --dataset sold \
    --intermediate mrp \
    --mask_ratio "${MASK_RATIO}" \
    --seed "${SEED}" \
    --val_int 10000 \
    --patience 3 \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.00002 \
    --n_tk_label 2 \
    --skip_empty_rat True \
    --push_to_hub True \
    --short_name True

# MRP Testing for epochs 3-5
for epoch in {3..5}; do
    python main.py \
        --pretrained_model "${BASE_MODEL}" \
        --wandb_p "${wandb_p}" \
        --exp_save_name "${COMMON_EXP_NAME}_pre" \
        --test_model_path "pre_finetune/${COMMON_EXP_NAME}_pre/ep${epoch}.ckpt" \
        --finetuning_stage pre \
        --dataset sold \
        --intermediate mrp \
        --mask_ratio "${MASK_RATIO}" \
        --seed "${SEED}" \
        --test True \
        --val_int 10000 \
        --patience 3 \
        --epochs 1 \
        --batch_size 1 \
        --n_tk_label 2 \
        --skip_empty_rat True \
        --push_to_hub True \
        --short_name True
done

############################################
## Stage 2: HS (Final Stage)
############################################

# Combined the three identical training blocks into a loop
# CHANGE: Consolidated repeated code into a single loop
for ep in {3..5}; do
    python main.py \
        --pretrained_model "${BASE_MODEL}" \
        --wandb_p "${wandb_p}" \
        --exp_save_name "${COMMON_EXP_NAME}_final_ep${ep}" \
        --pre_finetuned_model "pre_finetune/${COMMON_EXP_NAME}_pre/ep${ep}.ckpt" \
        --finetuning_stage final \
        --dataset sold \
        --mask_ratio "${MASK_RATIO}" \
        --seed "${SEED}" \
        --val_int 10000 \
        --patience 3 \
        --epochs 5 \
        --batch_size 16 \
        --lr 0.00002 \
        --num_labels 2 \
        --push_to_hub True \
        --short_name True
done

# Combined the three identical testing blocks into a loop
# CHANGE: Consolidated repeated testing code into a single loop
for pre_ep in {3..5}; do
    for epoch in {3..5}; do
        python main.py \
            --pretrained_model "${BASE_MODEL}" \
            --wandb_p "${wandb_p}" \
            --exp_save_name "${COMMON_EXP_NAME}_final" \
            --test_model_path "final_finetune/${COMMON_EXP_NAME}_final_ep${pre_ep}/ep${epoch}.ckpt" \
            --finetuning_stage final \
            --dataset sold \
            --mask_ratio "${MASK_RATIO}" \
            --seed "${SEED}" \
            --test True \
            --val_int 10000 \
            --patience 3 \
            --epochs 1 \
            --batch_size 1 \
            --lr 0.00002 \
            --num_labels 2 \
            --push_to_hub True \
            --short_name True
    done
done
