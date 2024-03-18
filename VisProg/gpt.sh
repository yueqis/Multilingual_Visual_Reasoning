#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate visprog

LAN=zh
# TEST_FILE=/home/yueqis/marvl/data/en/dev.csv
TEST_FILE=/home/yueqis/marvl/data/${LAN}/annotations_machine-translate/marvl-${LAN}_gmt.csv
OUTPUT_FILE=/home/yueqis/marvl/visprog/outputs/gpt3.5/test/${LAN}.csv
API_KEY=sk-abcdefgh # put your openai api key here

python3 nlvr_gpt.py \
    --test_file ${TEST_FILE} \
    --output_file ${OUTPUT_FILE} \
    --api_key ${API_KEY} \
