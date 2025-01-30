# NO MRP - without Augmented Data - seed 42

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project baseline-without-augmented-2-seed42-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model xlm-roberta-base --exp_save_name final_wo_mrp_0.5_seed42_no_ckpy --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --wandb_project baseline-without-augmented-2-seed42-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed42_no_ckpy/ep3.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --wandb_project baseline-without-augmented-2-seed42-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed42_no_ckpy/ep4.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --wandb_project baseline-without-augmented-2-seed42-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed42_no_ckpy/ep5.ckpt --push_to_hub True

# Next Seed

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 52 --wandb_project baseline-without-augmented-2-seed52-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model xlm-roberta-base --exp_save_name final_wo_mrp_0.5_seed52_no_ckpy --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 52 --wandb_project baseline-without-augmented-2-seed52-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed52_no_ckpy/ep3.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 52 --wandb_project baseline-without-augmented-2-seed52-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed52_no_ckpy/ep4.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 52 --wandb_project baseline-without-augmented-2-seed52-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed52_no_ckpy/ep5.ckpt --push_to_hub True

# Next Seed

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 62 --wandb_project baseline-without-augmented-2-seed62-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model xlm-roberta-base --exp_save_name final_wo_mrp_0.5_seed62_no_ckpy --short_name True --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 62 --wandb_project baseline-without-augmented-2-seed62-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed62_no_ckpy/ep3.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 62 --wandb_project baseline-without-augmented-2-seed62-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed62_no_ckpy/ep4.ckpt --push_to_hub True

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 62 --wandb_project baseline-without-augmented-2-seed62-no-pre --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --test_model_path final_finetune/final_wo_mrp_0.5_seed62_no_ckpy/ep5.ckpt --push_to_hub True