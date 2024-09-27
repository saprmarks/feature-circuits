#!/bin/bash

MODEL=$1
CIRCUIT=$2
EVAL_DATA=$3
THRESHOLD=$4
LENGTH=$5
DICTID=$6

# Run the ablation.py script with the specified arguments
python ablation.py \
--model $MODEL \
--dict_path sae_lens \
--circuit $CIRCUIT \
--data ${EVAL_DATA}.json \
--num_examples 40 \
--length $LENGTH \
--dict_id $DICTID \
--dict_size 16384 \
--threshold $THRESHOLD \
--ablation mean \
--handle_errors 'default' \
--start_layer 2 \
--device cuda:0