#!/bin/bash
source /nethome/rramrakhya6/miniconda3/etc/profile.d/conda.sh
conda activate habitat

url=$1
filename=$2
output=$3
maxEpLength=$4
echo "Downloading episodes.."
wget $url

unzip -o $filename

echo "Starting episode parsing!"
python psiturk_dataset/parsing/parser.py --replay-path data/hit_data/visualisation/unapproved_hits/ --output-path $output --max-episode-length $maxEpLength

python psiturk_dataset/parsing/check_train_val_leak --input-path-1 $output --list-duplicate --write-deduped $output
