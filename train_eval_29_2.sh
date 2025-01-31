# # # MRP and RP Experiments for Seed = 42 | Stage 1 of Finetuning Strategy

# # # This script is for running the training experiment for the first stage of the finetuning strategy for MRP with mask ratio 0.5 for seed 42.
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.5 --push_to_hub True --short_name True --exp_save_name stage1-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5 --wandb_project xlmr-base-stage1-29th

# # # This script is for running the training experiment for the first stage of the finetuning strategy for MRP with mask ratio 0.75 for seed 42.
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.75 --push_to_hub True --short_name True --exp_save_name stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75 --wandb_project xlmr-base-stage1-29th

# # # This script is for running the training experiment for the first stage of the finetuning strategy for MRP with mask ratio 0.9 for seed 42.
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.9 --push_to_hub True --short_name True --exp_save_name stage1-mrp_0.9_seed42_b16_e5_radam_s42_msk0.9 --wandb_project xlmr-base-stage1-29th

# # # This script is for running the training experiment for the first stage of the finetuning strategy for MRP with mask ratio 0.25 for seed 42.
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate mrp --mask_ratio 0.25 --push_to_hub True --short_name True --exp_save_name stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25 --wandb_project xlmr-base-stage1-29th

# # # This script is for running the training experiment for the first stage of the finetuning strategy for RP with mask ratio 1.0 for seed 42.
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage pre --dataset sold --n_tk_label 2 --skip_empty_rat True --intermediate rp --mask_ratio 1.0 --push_to_hub True --short_name True --exp_save_name stage1-rp_1.0_seed42_b16_e5_radam_s42_msk1.0 --wandb_project xlmr-base-stage1-29th

# # # Stage 2 of Finetuning Strategy for Seed 42

# # # This script is for running the training experiment for the second stage of the finetuning strategy for MRP with mask ratio 0.5 for seed 42 (Loads the pre-finetuned model from the first stage, using the checkpoint from last epoch).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --short_name True --push_to_hub True --intermediate mrp --pre_finetuned_model pre_finetune/stage1-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5/ep5.ckpt --exp_save_name stage2-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2_ep5_ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the training experiment for the second stage of the finetuning strategy for MRP with mask ratio 0.75 for seed 42 (Loads the pre-finetuned model from the first stage, using the checkpoint from last epoch).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.75 --short_name True --push_to_hub True --intermediate mrp --pre_finetuned_model pre_finetune/stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75/ep5.ckpt --exp_save_name stage2-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75_2_ep5_ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the training experiment for the second stage of the finetuning strategy for MRP with mask ratio 0.9 for seed 42 (Loads the pre-finetuned model from the first stage, using the checkpoint from last epoch).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.9 --short_name True --push_to_hub True --intermediate mrp --pre_finetuned_model pre_finetune/stage1-mrp_0.9_seed42_b16_e5_radam_s42_msk0.9/ep5.ckpt --exp_save_name stage2-mrp_0.9_seed42_b16_e5_radam_s42_msk0.9_2_ep5_ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the training experiment for the second stage of the finetuning strategy for MRP with mask ratio 0.25 for seed 42 (Loads the pre-finetuned model from the first stage, using the checkpoint from last epoch).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.25 --short_name True --push_to_hub True --intermediate mrp --pre_finetuned_model pre_finetune/stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25/ep5.ckpt --exp_save_name stage2-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25_2_ep5_ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the training experiment for the second stage of the finetuning strategy for RP with mask ratio 1.0 for seed 42 (Loads the pre-finetuned model from the first stage, using the checkpoint from last epoch).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 5 --batch_size 16 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 1.0 --short_name True --push_to_hub True --intermediate rp --pre_finetuned_model pre_finetune/stage1-rp_1.0_seed42_b16_e5_radam_s42_msk1.0/ep5.ckpt --exp_save_name stage2-rp_1.0_seed42_b16_e5_radam_s42_msk1.0_2_ep5_ckpt --wandb_project xlmr-base-stage2-29th

# # # TODO : Testing experiments for Seed 42 of final stage of finetuning strategy

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.5 for seed 42 (Loads the model from the 4th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2_ep5_ckpt/ep4.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.5 for seed 42 (Loads the model from the 5th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.5 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5_2_ep5_ckpt/ep5.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.75 for seed 42 (Loads the model from the 4th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.75 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75_2_ep5_ckpt/ep4.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.75 for seed 42 (Loads the model from the 5th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.75 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75_2_ep5_ckpt/ep5.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.9 for seed 42 (Loads the model from the 5th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.9 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.9_seed42_b16_e5_radam_s42_msk0.9_2_ep5_ckpt/ep5.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for MRP with mask ratio 0.25 for seed 42 (Loads the model from the 5th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 0.25 --test True --push_to_hub True --intermediate mrp --test_model_path final_finetune/stage2-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25_2_ep5_ckpt/ep5.ckpt --wandb_project xlmr-base-stage2-29th

# # # This script is for running the testing experiment for the final stage of the finetuning strategy for RP with mask ratio 1.0 for seed 42 (Loads the model from the 5th epoch of the final stage of finetuning strategy).
# # python main.py --pretrained_model xlm-roberta-base --val_int 10000 --patience 3 --epochs 1 --batch_size 1 --lr 0.00002 --seed 42 --finetuning_stage final --dataset sold --num_labels 2 --mask_ratio 1.0 --test True --push_to_hub True --intermediate rp --test_model_path final_finetune/stage2-rp_1.0_seed42_b16_e5_radam_s42_msk1.0_2_ep5_ckpt/ep5.ckpt --wandb_project xlmr-base-stage2-29th

# # TODO : Training experiments for Seed 42 of final stage of finetuning strategy without MRP

# # 
# # Define the seeds : seeds=(42 52 62 72 82)
# # seeds=(42)

# # # Loop through each seed
# # for seed in "${seeds[@]}"; do
# #     echo "Running experiments with seed: $seed"
    
# #     # Training experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 5 \
# #         --batch_size 16 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio -1 \
# #         --short_name True \
# #         --push_to_hub True \
# #         --intermediate none \
# #         --pre_finetuned_model xlm-roberta-base \
# #         --exp_save_name "stage2-none_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt" \
# #         --wandb_project xlmr-base-stage2-none-29th

# #     # Testing experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 1 \
# #         --batch_size 1 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio -1 \
# #         --test True \
# #         --push_to_hub True \
# #         --intermediate none \
# #         --test_model_path "final_finetune/stage2-none_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt/ep5.ckpt" \
# #         --wandb_project xlmr-base-stage2-none-29th

# #     echo "Completed experiments with seed: $seed"
# # done

# # rf -rm final_finetune/stage2-none_seed42_b16_e5_radam_s42_msk1.0_2_ep5_ckpt
# # # Define the seeds : 
# # seeds=(52 62 72 82)
# # # Loop through each seed
# # for seed in "${seeds[@]}"; do
# #     echo "Running experiments with seed: $seed"
    
# #     # Training experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 5 \
# #         --batch_size 16 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio -1 \
# #         --short_name True \
# #         --push_to_hub True \
# #         --intermediate none \
# #         --pre_finetuned_model xlm-roberta-base \
# #         --exp_save_name "stage2-none_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt" \
# #         --wandb_project xlmr-base-stage2-none-29th

# #     # Testing experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 1 \
# #         --batch_size 1 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio -1 \
# #         --test True \
# #         --push_to_hub True \
# #         --intermediate none \
# #         --test_model_path "final_finetune/stage2-none_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt/ep5.ckpt" \
# #         --wandb_project xlmr-base-stage2-none-29th

# #     echo "Completed experiments with seed: $seed"

# #     rm -rf final_finetune/stage2-none_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt
# # done

# # # Define the random seeds
# # seeds=(52 62 72 82 42)

# # # Loop through each seed
# # for seed in "${seeds[@]}"; do
# #     echo "Running experiments with seed: $seed"

# #     # Training (Stage 1) experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 5 \
# #         --batch_size 16 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage pre \
# #         --dataset sold \
# #         --n_tk_label 2 \
# #         --skip_empty_rat True \
# #         --intermediate mrp \
# #         --mask_ratio 0.5 \
# #         --push_to_hub True \
# #         --short_name True \
# #         --exp_save_name "stage1-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5" \
# #         --wandb_project xlmr-base-stage1-29th-more_seeds

# #     # Fine-tuning (Stage 2) experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 5 \
# #         --batch_size 16 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio 0.5 \
# #         --short_name True \
# #         --push_to_hub True \
# #         --intermediate mrp \
# #         --pre_finetuned_model "pre_finetune/stage1-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5/ep5.ckpt" \
# #         --exp_save_name "stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt" \
# #         --wandb_project xlmr-base-stage2-29th-more_seeds

# #     # Testing experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 1 \
# #         --batch_size 1 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio 0.5 \
# #         --test True \
# #         --push_to_hub True \
# #         --intermediate mrp \
# #         --test_model_path "final_finetune/stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt/ep5.ckpt" \
# #         --wandb_project xlmr-base-stage2-29th-more_seeds

# #     echo "Completed experiments with seed: $seed"

# #     rm -rf pre_finetune/stage1-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5
# #     rm -rf final_finetune/stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt

# #     echo "Removed pre-finetuned and fine-tuned models for seed: $seed"
# # done

# # # Define the random seeds
# # seeds=(52 62 72 82 42)

# # # Loop through each seed
# # for seed in "${seeds[@]}"; do
# #     echo "Running experiments with seed: $seed"

# #     # Testing experiment
# #     python main.py --pretrained_model xlm-roberta-base \
# #         --val_int 10000 \
# #         --patience 3 \
# #         --epochs 1 \
# #         --batch_size 1 \
# #         --lr 0.00002 \
# #         --seed $seed \
# #         --finetuning_stage final \
# #         --dataset sold \
# #         --num_labels 2 \
# #         --mask_ratio 0 \
# #         --test True \
# #         --push_to_hub False \
# #         --intermediate none \
# #         --test_model_path "xlm-roberta-base" \
# #         --wandb_project xlmr-base-stage2-29th-no_training_atall

# #     echo "Completed experiments with seed: $seed"

# # done

# # Define the random seeds
# seeds=(52 62 72 82 42)

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed: $seed"

#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 1.0 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate rp \
#         --pre_finetuned_model "pre_finetune/stage1-rp_1.0_seed42_b16_e5_radam_s42_msk1.0/ep5.ckpt" \
#         --exp_save_name "stage2-rp_1.0_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-rp1.0

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 1.0 \
#         --test True \
#         --push_to_hub True \
#         --intermediate rp \
#         --test_model_path "final_finetune/stage2-rp_1.0_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-rp1.0

#     echo "Completed experiments with seed: $seed"

#     rm -rf final_finetune/stage2-rp_1.0_seed${seed}_b16_e5_radam_s${seed}_msk1.0_2_ep5_ckpt

#     echo "Removed d fine-tuned models for seed: $seed"
# done

# # Define the random seeds
# seeds=(52 62 72 82 42)

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed: $seed"

#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75/ep5.ckpt" \
#         --exp_save_name "stage2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

#     echo "Completed experiments with seed: $seed"

#     rm -rf final_finetune/stage2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt

#     echo "Removed pre-finetuned and fine-tuned models for seed: $seed"
# done

# # Define the random seeds
# seeds=(52 62 72 82 42)

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed: $seed"

#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.25 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25/ep5.ckpt" \
#         --exp_save_name "stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

#     echo "Completed experiments with seed: $seed"

#     rm -rf final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt

#     echo "Removed pre-finetuned and fine-tuned models for seed: $seed"
# done





# ## AUGMENTED DATA ##

# # MR - 0.5 Augmented Dataset
# seeds=(52 62 72 82 42)

# # Training (Stage 1) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed 42 \
#         --finetuning_stage pre \
#         --dataset sold \
#         --n_tk_label 2 \
#         --skip_empty_rat True \
#         --intermediate mrp \
#         --mask_ratio 0.5 \
#         --push_to_hub True \
#         --short_name True \
#         --exp_save_name "stage1-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5" \
#         --wandb_project xlmr-base-stage1-29th-msk0.5-augmented
# 		--use_augmented_dataset True

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed with Augmented Data: $seed"

   
#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.5_seed42_b16_e5_radam_s42_msk0.5/ep5.ckpt" \
#         --exp_save_name "stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.5-augmented

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.5 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.5-augmented

#     echo "Completed experiments with seed: $seed"
	
#     rm -rf final_finetune/stage2-mrp_0.5_seed${seed}_b16_e5_radam_s${seed}_msk0.5_2_ep5_ckpt

#     echo "Removed fine-tuned models for seed: $seed"
# done


# # MR - 0.75 Augmented Dataset
# seeds=(52 62 72 82 42)

# # Training (Stage 1) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed 42 \
#         --finetuning_stage pre \
#         --dataset sold \
#         --n_tk_label 2 \
#         --skip_empty_rat True \
#         --intermediate mrp \
#         --mask_ratio 0.75 \
#         --push_to_hub True \
#         --short_name True \
#         --exp_save_name "stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75" \
#         --wandb_project xlmr-base-stage1-29th-msk0.75-augmented
# 		--use_augmented_dataset True

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed with Augmented Data: $seed"

   
#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75/ep5.ckpt" \
#         --exp_save_name "stage2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented

#     echo "Completed experiments with seed: $seed"
	
#     rm -rf final_finetune/stage2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt

#     echo "Removed fine-tuned models for seed: $seed"
# done

# # MR - 0.25 Augmented Dataset
# seeds=(52 62 72 82 42)

# # Training (Stage 1) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed 42 \
#         --finetuning_stage pre \
#         --dataset sold \
#         --n_tk_label 2 \
#         --skip_empty_rat True \
#         --intermediate mrp \
#         --mask_ratio 0.25 \
#         --push_to_hub True \
#         --short_name True \
#         --exp_save_name "stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25" \
#         --wandb_project xlmr-base-stage1-29th-msk0.25-augmented
# 		--use_augmented_dataset True

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed with Augmented Data: $seed"

   
#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.25 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25/ep5.ckpt" \
#         --exp_save_name "stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.25-augmented

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.25 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.25-augmented

#     echo "Completed experiments with seed: $seed"
	
#     rm -rf final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt

#     echo "Removed fine-tuned models for seed: $seed"
# done




# MR - 0.75 Augmented Dataset
seeds=(42 52 62 72 82)

# # Training (Stage 1) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed 42 \
#         --finetuning_stage pre \
#         --dataset sold \
#         --n_tk_label 2 \
#         --skip_empty_rat True \
#         --intermediate mrp \
#         --mask_ratio 0.75 \
#         --push_to_hub True \
#         --short_name True \
#         --exp_save_name "stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75-sold-aug" \
#         --wandb_project xlmr-base-stage1-29th-msk0.75-augmented-new \
#         --use_augmented_dataset True \
#         --max_gen_per_sample 1

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments with seed with Augmented Data: $seed"

   
#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75-sold-aug/ep5.ckpt" \
#         --exp_save_name "stage2-sold-aug-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented-new \
#         --use_augmented_dataset True

#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-sold-aug-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented-new

#     echo "Completed experiments with seed: $seed"
	
#     rm -rf final_finetune/stage2-sold-aug-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt

#     echo "Removed fine-tuned models for seed: $seed"
# done

# # Training (Stage 1) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed 42 \
#         --finetuning_stage pre \
#         --dataset sold \
#         --n_tk_label 2 \
#         --skip_empty_rat True \
#         --intermediate mrp \
#         --mask_ratio 0.75 \
#         --push_to_hub True \
#         --short_name True \
#         --exp_save_name "stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75-sold-aug-gen2" \
#         --wandb_project xlmr-base-stage1-29th-msk0.75-augmented-new \
#         --use_augmented_dataset True \
#         --max_gen_per_sample 2

# # MR - 0.75 Augmented Dataset with 2 generations per sample
# seeds=(42 52 62 72 82)

# # Loop through each seed
# for seed in "${seeds[@]}"; do
#     echo "Running experiments (2 new gens) with seed with Augmented Data: $seed"

#     # Fine-tuning (Stage 2) experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 5 \
#         --batch_size 16 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --short_name True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --pre_finetuned_model "pre_finetune/stage1-mrp_0.75_seed42_b16_e5_radam_s42_msk0.75-sold-aug-gen2/ep5.ckpt" \
#         --exp_save_name "stage2-sold-aug-gen2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented-new-2gen \
#         --use_augmented_dataset True \
#         --max_gen_per_sample 2
    
#     # Testing experiment
#     python main.py --pretrained_model xlm-roberta-base \
#         --val_int 10000 \
#         --patience 3 \
#         --epochs 1 \
#         --batch_size 1 \
#         --lr 0.00002 \
#         --seed $seed \
#         --finetuning_stage final \
#         --dataset sold \
#         --num_labels 2 \
#         --mask_ratio 0.75 \
#         --test True \
#         --push_to_hub True \
#         --intermediate mrp \
#         --test_model_path "final_finetune/stage2-sold-aug-gen2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt/ep5.ckpt" \
#         --wandb_project xlmr-base-stage2-29th-msk0.75-augmented-new-2gen

#     echo "Completed experiments with seed: $seed"

#     rm -rf final_finetune/stage2-sold-aug-gen2-mrp_0.75_seed${seed}_b16_e5_radam_s${seed}_msk0.75_2_ep5_ckpt

#     echo "Removed fine-tuned models for seed: $seed"

# done


# Define the random seeds
seeds=(42) # 52 62 72 82 

# Loop through each seed
for seed in "${seeds[@]}"; do
    echo "Running experiments with seed: $seed"

    # Fine-tuning (Stage 2) experiment
    python main.py --pretrained_model xlm-roberta-base \
        --val_int 10000 \
        --patience 3 \
        --epochs 5 \
        --batch_size 16 \
        --lr 0.00002 \
        --seed $seed \
        --finetuning_stage final \
        --dataset sold \
        --num_labels 2 \
        --mask_ratio 0.25 \
        --short_name True \
        --push_to_hub True \
        --intermediate mrp \
        --pre_finetuned_model "pre_finetune/stage1-mrp_0.25_seed42_b16_e5_radam_s42_msk0.25/ep5.ckpt" \
        --exp_save_name "stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt" \
        --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

    # Testing experiment
    python main.py --pretrained_model xlm-roberta-base \
        --val_int 10000 \
        --patience 3 \
        --epochs 1 \
        --batch_size 1 \
        --lr 0.00002 \
        --seed $seed \
        --finetuning_stage final \
        --dataset sold \
        --num_labels 2 \
        --mask_ratio 0.5 \
        --test True \
        --push_to_hub True \
        --intermediate mrp \
        --test_model_path "final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt/ep5.ckpt" \
        --wandb_project xlmr-base-stage2-29th-more_seeds-mrp-0.75

    echo "Completed experiments with seed: $seed"

    rm -rf final_finetune/stage2-mrp_0.25_seed${seed}_b16_e5_radam_s${seed}_msk0.25_2_ep5_ckpt

    echo "Removed pre-finetuned and fine-tuned models for seed: $seed"
done