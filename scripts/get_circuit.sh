#!/bin/bash

DATA=$1
NODE=$2
EDGE=$3
AGG=$4
LENGTH=$5
DICT_ID=$6

python circuit.py \
    --model EleutherAI/pythia-70m-deduped \
    --num_examples 100 \
    --batch_size 10 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --example_length $LENGTH \
    --dict_id $DICT_ID