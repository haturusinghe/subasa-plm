# Baseline - With Augmented Data

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project baseline-with-augmented-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --push_to_hub True --exp_save_name mrp_0.5_seed42_aug --short_name True --use_augmented_dataset True

python main.py --pretrained_model xlm-roberta-base --val_int 25000 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project baseline-with-augmented-1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.5 --test_model_path pre_finetune/mrp_0.5_seed42_aug/ep4.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project baseline-with-augmented-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/mrp_0.5_seed42_aug/ep4.ckpt --exp_save_name final_w_mrp_0.5_seed42_aug --short_name True --push_to_hub True --use_augmented_dataset True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --wandb_project baseline-with-augmented-1 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_w_mrp_0.5_seed42_aug/ep4.ckpt --push_to_hub True