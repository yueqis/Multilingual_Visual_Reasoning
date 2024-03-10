#!/bin/bash

conda activate uniter

LAN=tr
TASK=12
MODEL=ctrl_xuniter
MODEL_CONFIG=ctrl_xuniter_base
TRTASK=NLVR2
TETASK=MaRVL${LAN}
TASKS_CONFIG=xling_test_marvl
TEXT_PATH=/Multilingual_Visual_Reasoning/data/${LAN}/annotations/marvl-${LAN}.jsonl
FEAT_PATH=/Multilingual_Visual_Reasoning/uniters/feature_extraction/ERDA/marvl-${LAN}_boxes36.lmdb
PRETRAINED=/Multilingual_Visual_Reasoning/uniters/results/${MODEL}/NLVR2/train/NLVR2_${MODEL}_base/pytorch_model_best.bin
OUTPUT_DIR=/Multilingual_Visual_Reasoning/uniters/results/${MODEL}/${LAN}

cd ../../../volta
python3 eval_task.py \
        --bert_model xlm-roberta-base \
        --config_file config/${MODEL_CONFIG}.json \
        --from_pretrained ${PRETRAINED} \
        --val_annotations_jsonpath ${TEXT_PATH} --val_features_lmdbpath ${FEAT_PATH} \
        --tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK --split test \
        --output_dir ${OUTPUT_DIR} 