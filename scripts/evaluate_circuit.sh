#!/bin/bash

MODEL=$1
CIRCUIT=$2
EVAL_DATA=$3
THRESHOLD=$4
START_LAYER=$5


# Run the ablation.py script with the specified arguments
python ablation.py \
--model $MODEL \
--circuit $CIRCUIT \
--data ${EVAL_DATA}.json \
--num_examples 40 \
--dict_id $DICTID \
--threshold $THRESHOLD \
--ablation mean \
--handle_errors 'default' \
--start_layer $START_LAYER \
--batch_size 20 \
--device cuda:0