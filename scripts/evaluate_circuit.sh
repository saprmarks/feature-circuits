#!/bin/bash

CIRCUIT=$1
EVAL_DATA=$2
THRESHOLD=$3
LENGTH=$4
DICTID=$5

# Run the ablation.py script with the specified arguments
python ablation.py \
--dict_path dictionaries/pythia-70m-deduped/ \
--circuit $CIRCUIT \
--data ${EVAL_DATA}.json \
--num_examples 40 \
--length $LENGTH \
--dict_id $DICTID \
--dict_size 32768 \
--threshold $THRESHOLD \
--ablation mean \
--handle_errors 'default' \
--start_layer 2 \
--device cuda:0