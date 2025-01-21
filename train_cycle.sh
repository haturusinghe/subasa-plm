# Baseline

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --push_to_hub True --exp_save_name mrp_0.5_seed43 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.5 --test_model_path pre_finetune/mrp_0.5_seed43/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/mrp_0.5_seed43/ep4.ckpt --exp_save_name final_w_mrp_0.5_seed43 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_w_mrp_0.5_seed43/ep4.ckpt --push_to_hub True

# Mask Ratio 0.1

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.1 --push_to_hub True --exp_save_name mrp_0.1_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.1 --test_model_path pre_finetune/mrp_0.1_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.1 --pre_finetuned_model pre_finetune/mrp_0.1_seed66/ep4.ckpt --exp_save_name final_w_mrp_0.1_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.1 --test True --test_model_path final_finetune/final_w_mrp_0.1_seed66/ep4.ckpt --push_to_hub True

# Mask Ratio 0.25

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.25 --push_to_hub True --exp_save_name mrp_0.25_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.25 --test_model_path pre_finetune/mrp_0.25_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.25 --pre_finetuned_model pre_finetune/mrp_0.25_seed66/ep4.ckpt --exp_save_name final_w_mrp_0.25_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.25 --test True --test_model_path final_finetune/final_w_mrp_0.25_seed66/ep4.ckpt --push_to_hub True

# Mask Ratio 0.75

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.75 --push_to_hub True --exp_save_name mrp_0.75_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.75 --test_model_path pre_finetune/mrp_0.75_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.75 --pre_finetuned_model pre_finetune/mrp_0.75_seed66/ep4.ckpt --exp_save_name final_w_mrp_0.75_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.75 --test True --test_model_path final_finetune/final_w_mrp_0.75_seed66/ep4.ckpt --push_to_hub True

# Mask Ratio 0.9

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.9 --push_to_hub True --exp_save_name mrp_0.9_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.9 --test_model_path pre_finetune/mrp_0.9_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.9 --pre_finetuned_model pre_finetune/mrp_0.9_seed66/ep4.ckpt --exp_save_name final_w_mrp_0.9_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.9 --test True --test_model_path final_finetune/final_w_mrp_0.9_seed66/ep4.ckpt --push_to_hub True

# Mask Ratio 1 - RP

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate rp --mask_ratio 1 --push_to_hub True --exp_save_name rp_1_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate rp --test True --mask_ratio 1 --test_model_path pre_finetune/rp_1_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 1 --pre_finetuned_model pre_finetune/rp_1_seed66/ep4.ckpt --exp_save_name final_w_rp_1_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 1 --test True --test_model_path final_finetune/final_w_rp_1_seed66/ep4.ckpt --push_to_hub True
