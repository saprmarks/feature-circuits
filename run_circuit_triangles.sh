#!/bin/bash

DATA=$1
NODE=$2
EDGE=$3
AGG=$4
DICT_ID=$5

python circuit_triangles.py \
    --model EleutherAI/pythia-70m-deduped \
    --num_examples 100 \
    --batch_size 10 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --example_length 6 \
    --dict_id $DICT_ID

# --submodules model.gpt_neox.layers.{}.attention.dense, model.gpt_neox.layers.{}.mlp.dense_4h_to_h,model.gpt_neox.layers.{}
