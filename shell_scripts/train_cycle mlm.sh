# Mask Ratio 0.15 - MLM

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mlm --mask_ratio 0.15 --push_to_hub True --exp_save_name mlm_0.15_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mlm --test True --mask_ratio 0.15 --test_model_path pre_finetune/mlm_0.15_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.15 --pre_finetuned_model pre_finetune/mlm_0.15_seed66/ep4.ckpt --exp_save_name final_w_mlm_0.15_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.15 --test True --test_model_path final_finetune/final_w_mlm_0.15_seed66/ep4.ckpt --push_to_hub True

# Mask Ratio 0.5 - MLM

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mlm --mask_ratio 0.5 --push_to_hub True --exp_save_name mlm_0.5_seed66 --short_name True

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mlm --test True --mask_ratio 0.5 --test_model_path pre_finetune/mlm_0.5_seed66/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/mlm_0.5_seed66/ep4.ckpt --exp_save_name final_w_mlm_0.5_seed66 --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 1000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 66 --wandb_project ablataion-study-ctrl-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_w_mlm_0.5_seed66/ep4.ckpt --push_to_hub True

# STOP GOOGLE COLLAB SESSION VIA COMMAND LINE
python -c "from google.colab import runtime; runtime.unassign()"