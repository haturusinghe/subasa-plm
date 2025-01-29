python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project xlmr-base-stage1-42-b16-e5-radam-s42-mrp-msk0.5 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --push_to_hub True --short_name True --exp_save_name mrp_0.5_seed42_b16_e5_radam_s42_msk0.5

# python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project xlmr-base-42-b16-e5-radam-s42-mrp-msk0.5 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --test True --test_model_path pre_finetune/mrp_0.5_seed42_b16_e5_radam_s42_msk0.5/ep5.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project xlmr-base-42-b16-e5-radam-s42-mrp-msk0.5 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --test True --test_model_path pre_finetune/mrp_0.5_seed42_b16_e5_radam_s42_msk0.5/ep5.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 25000 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project baseline-without-augmented-2-seed42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.5 --test_model_path pre_finetune/mrp_0.5_seed42/ep3.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project xlmr-base-stage2-42-b16-e5-radam-s42-mrp-msk0.5 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2/ep5.ckpt --exp_save_name mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2_ep5_ckpt --short_name True --push_to_hub True --intermediate mrp

main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --wandb_project xlmr-base-42-b16-e5-radam-s42-mrp-msk0.5 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --pre_finetuned_model pre_finetune/mrp_0.5_seed42_b16_e5_radam_s42_msk0.5/ep5.ckpt --exp_save_name mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2_ep5_ckpt --short_name True --push_to_hub True --intermediate mrp


