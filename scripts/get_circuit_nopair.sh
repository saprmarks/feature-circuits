#!/bin/bash

MODEL=$1
DATA=$2 # path/to/cluster/data
NODE=$3
EDGE=$4

python circuit.py \
    --model $MODEL \
    --num_examples 100 \
    --batch_size 6 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation sum \
	--nopair
