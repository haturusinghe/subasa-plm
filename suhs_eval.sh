# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "s-haturusinghe/final_w_mrp_0.75_seed66_ep4" \
#         --wandb_project best-subasa-scores

# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset suhs \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "s-haturusinghe/final_w_mrp_0.75_seed66_ep4" \
#         --wandb_project suhs-plm-eval

# for s-haturusinghe/stage2-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75_2_ep5_ckpt_final_ep5

# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset suhs \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "s-haturusinghe/stage2-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75_2_ep5_ckpt_final_ep5" \
#         --wandb_project suhs-plm-eval

# # for s-haturusinghe/stage2-none_seed52_b16_e5_radam_s52_msk1.0_2_ep5_ckpt_final_ep5
# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset suhs \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "s-haturusinghe/stage2-none_seed52_b16_e5_radam_s52_msk1.0_2_ep5_ckpt_final_ep5" \
#         --wandb_project suhs-plm-eval

# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "keshan/sinhala-roberta-oscar" \
#         --wandb_project other-baselines-against-subasa-xlmr-new1

# python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed 66 \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "keshan/SinhalaBERTo" \
#         --wandb_project other-baselines-against-subasa-xlmr-new1

# NLPC-UOM/SinBERT-large

python main.py --pretrained_model xlm-roberta-base \
        --val_int 10000 \
        --patience 3 \
        --epochs 1 \
        --batch_size 1 \
        --lr 0.00002 \
        --seed 66 \
        --finetuning_stage final \
        --dataset sold \
        --num_labels 2 \
        --mask_ratio 0.5 \
        --test True \
        --push_to_hub True \
        --intermediate mrp \
        --test_model_path "NLPC-UOM/SinBERT-large" \
        --wandb_project other-baselines-against-subasa-xlmr-new1