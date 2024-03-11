#!/bin/bash

#SBATCH --job-name=gpt4v-id
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -c 8
#SBATCH --time 2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yueqis@andrew.cmu.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate visprog

api_key_neulab=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

language=id
# test_file=/home/yueqis/marvl/data/en/dev.csv
# output_file=/home/yueqis/marvl/gpt4v/marvl/output/marvl/${language}.csv
# test_file=/home/yueqis/marvl/data/${language}/annotations/marvl-${language}.csv
# output_file=/home/yueqis/marvl/gpt4v/marvl/output/marvl/${language}2.csv
test_file=/home/yueqis/marvl/data/${language}/annotations_machine-translate/marvl-${language}_gmt.csv
output_file=/home/yueqis/marvl/gpt4v/marvl/output/translate/${language}2.csv
start=0
length=10000

python3 gpt4v.py --test_file $test_file --output_file $output_file --api_key $api_key_neulab --start $start --length $length
