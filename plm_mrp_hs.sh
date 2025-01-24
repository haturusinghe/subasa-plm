# MRP -> HS | Mask Ratio for MRP = 0.5 , Seed = 13 | Wandb Project Name 
## MRP Training and Testing
python main.py \
    --pretrained_model xlm-roberta-base \
    --val_int 10000 \
    --patience 3 \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.00002 \
    --seed 13 \
    --wandb_project "sold_masking_pretraining" \
    --finetuning_stage pre \
    --dataset sold \
    --n_tk_label 2 \
    --skip_empty_rat True \
    --intermediate mrp \
    --mask_ratio 0.5 \
    --push_to_hub True \
    --exp_save_name "xlmr-sold-mrp_msk0.5_s13" \
    --short_name True
# Testing MRP checkpoint at epoch 3
python main.py \
    --pretrained_model xlm-roberta-base \
    --test True \
    --val_int 10000 \
    --patience 3 \
    --epochs 1 \
    --batch_size 1 \
    --seed 13 \
    --wandb_project "sold_masking_pretraining" \
    --finetuning_stage pre \
    --dataset sold \
    --n_tk_label 2 \
    --skip_empty_rat True \
    --intermediate mrp \
    --mask_ratio 0.5 \
    --push_to_hub True \
    --exp_save_name "xlmr-sold-mrp_msk0.5_s13" \
    --test_model_path  pre_finetune/xlmr-sold-mrp_msk0.5_s13/ep3.ckpt \
    --short_name True
# Testing MRP checkpoint at epoch 4
python main.py \
    --pretrained_model xlm-roberta-base \
    --test True \
    --val_int 10000 \
    --patience 3 \
    --epochs 1 \
    --batch_size 1 \
    --seed 13 \
    --wandb_project "sold_masking_pretraining" \
    --finetuning_stage pre \
    --dataset sold \
    --n_tk_label 2 \
    --skip_empty_rat True \
    --intermediate mrp \
    --mask_ratio 0.5 \
    --push_to_hub True \
    --exp_save_name "xlmr-sold-mrp_msk0.5_s13" \
    --test_model_path  pre_finetune/xlmr-sold-mrp_msk0.5_s13/ep4.ckpt \
    --short_name True 
# Testing MRP checkpoint at epoch 5
python main.py \
    --pretrained_model xlm-roberta-base \
    --test True \
    --val_int 10000 \
    --patience 3 \
    --epochs 1 \
    --batch_size 1 \
    --seed 13 \
    --wandb_project "sold_masking_pretraining" \
    --finetuning_stage pre \
    --dataset sold \
    --n_tk_label 2 \
    --skip_empty_rat True \
    --intermediate mrp \
    --mask_ratio 0.5 \
    --push_to_hub True \
    --exp_save_name "xlmr-sold-mrp_msk0.5_s13" \
    --test_model_path  pre_finetune/xlmr-sold-mrp_msk0.5_s13/ep5.ckpt \
    --short_name True 
# HS Training and Testing for pre-finetuned model with MRP Mask Ratio 0.5
python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 13 --wandb_project sold_masking_pretraining --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/xlmr-sold-mrp_msk0.5_s13/ep3.ckpt --short_name True --push_to_hub True --exp_save_name "xlmr-sold-final_mrp_msk0.5_s13"

python main.py --test True --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 13 --wandb_project sold_masking_pretraining --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 final_finetune/xlmr-sold-final_mrp_msk0.5_s13/ep3.ckpt --short_name True --push_to_hub True

python main.py --test True --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 13 --wandb_project sold_masking_pretraining --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 final_finetune/xlmr-sold-final_mrp_msk0.5_s13/ep4.ckpt --short_name True --push_to_hub True

python main.py --test True --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 13 --wandb_project sold_masking_pretraining --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 final_finetune/xlmr-sold-final_mrp_msk0.5_s13/ep5.ckpt --short_name True --push_to_hub True