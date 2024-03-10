#!/bin/bash

conda activate uniter

TASK=12
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TASKS_CONFIG=xling_test_nlvr2
TRTASK=NLVR2
TETASK=NLVR2
PRETRAINED=/Multilingual_Visual_Reasoning/uniters/results/${MODEL}/NLVR2/train/NLVR2_${MODEL}_base/pytorch_model_best.bin
OUTPUT_DIR=/Multilingual_Visual_Reasoning/uniters/results/${MODEL}/NLVR2/eval

cd ../../../volta
python3 eval_task.py \
        --bert_model xlm-roberta-base \
	--config_file config/${MODEL_CONFIG}.json \
        --from_pretrained ${PRETRAINED} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
        --output_dir ${OUTPUT_DIR}

