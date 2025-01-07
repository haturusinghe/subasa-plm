# Subasa - Adapting Language Models for Low Resourced Offensive Language Detection in Sinhala

A modular pipeline for training and evaluating XLM-RoBERTa models using PyTorch and Hugging Face Transformers.

python main.py --intermediate mrp --pretrained_model xlm-roberta-base --batch_size 16 --epochs 5 --lr 0.00005 --val_int 945 --patience 3 --mask_ratio 0.5 --n_tk_label 2

device = cuda
Checkpoint path:  pre_finetune/07012025-0936_LK_xlm-roberta_mrp_5e-05_16_945
Namespace(pretrained_model='xlm-roberta-base', batch_size=16, epochs=5, lr=5e-05, val_int=945, patience=3, intermediate='mrp', mask_ratio=0.5, n_tk_label=2, test=False, device=device(type='cuda'), exp_date='07012025-0936_LK', exp_name='07012025-0936_LK_xlm-roberta_mrp_5e-05_16_945', dir_result='pre_finetune/07012025-0936_LK_xlm-roberta_mrp_5e-05_16_945', waiting=0, n_eval=0)