python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project subasa-xlmr-base-session1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.1 --test_model_path pre_finetune/13012025-2347_LK_2e-05_16_1000_seed42_xlm-roberta-base_mrp_pre/xlm-roberta-base_pre_mrp_val_loss_0.106207_ep4_stp410_f1_0.609261__masked_f1_0.777782.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project subasa-xlmr-base-session1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.25 --test_model_path pre_finetune/14012025-0003_LK_2e-05_16_1000_seed42_xlm-roberta-base_mrp_pre/xlm-roberta-base_pre_mrp_val_loss_0.171203_ep4_stp410_f1_0.484313__masked_f1_0.484263.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project subasa-xlmr-base-session1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --test True --mask_ratio 0.75 --test_model_path pre_finetune/14012025-0019_LK_2e-05_16_1000_seed42_xlm-roberta-base_mrp_pre/xlm-roberta-base_pre_mrp_val_loss_0.062759_ep4_stp410_f1_0.609961__masked_f1_0.905189.ckpt

python main.py --pretrained_model xlm-roberta-base --val_int 2500 --patience 3 --epochs 1 --batch_size 1 --seed 42 --wandb_project subasa-xlmr-base-session1 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate rp --test True --mask_ratio 0.1 --test_model_path pre_finetune/14012025-0958_LK_2e-05_16_1000_seed42_xlm-roberta-base_rp_pre/xlm-roberta-base_pre_rp_val_loss_0.037872_ep4_stp410_f1_0.869262_.ckpt