#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate visprog

LAN=en
TEST_FILE=/home/yueqis/marvl/data/en/dev.csv
# TEST_FILE=/home/yueqis/marvl/data/${LAN}/annotations_machine-translate/marvl-${LAN}_gmt.csv
OUTPUT_FILE=/home/yueqis/marvl/visprog/outputs/llama-chat-70b/translate_test/${LAN}.csv
MODEL_ADDR=babel-4-36:9428 # put your own model address here

python3 nlvr_llama.py \
    --test_file ${TEST_FILE} \
    --output_file ${OUTPUT_FILE} \
    --model_addr ${MODEL_ADDR} \
    --sample_size 4 \