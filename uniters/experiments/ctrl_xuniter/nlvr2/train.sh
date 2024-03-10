#!/bin/bash

conda activate uniter

TASK=12
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=xling_trainval_nlvr2
PRETRAINED=/Multilingual_Visual_Reasoning/uniters/pretrained/${MODEL}/pytorch_model_9.bin
OUTPUT_DIR=/Multilingual_Visual_Reasoning/uniters/results/${MODEL}/NLVR2/train
LOGGING_DIR=/Multilingual_Visual_Reasoning/uniters/logs/${MODEL_CONFIG}

cd ../../../volta
python3 train_task.py \
        --bert_model xlm-roberta-base --config_file config/${MODEL_CONFIG}.json \
        --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --gradient_accumulation_steps 2 \
        --adam_epsilon 1e-6 --adam_betas 0.9 0.999 --adam_correct_bias --weight_decay 0.0001 --warmup_proportion 0.1 --clip_grad_norm 1.0 \
        --output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} 

