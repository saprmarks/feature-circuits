#!/bin/bash

MODEL=$1
DATA=$2
NODE=$3
EDGE=$4
AGG=$5

python circuit.py \
    --model $MODEL \
    --num_examples 100 \
    --batch_size 10 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG