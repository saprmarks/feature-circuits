#!/bin/bash

DATA=$1
NODE=$2
EDGE=$3
AGG=$4

python circuit_triangles.py \
    --model EleutherAI/pythia-70m-deduped \
    --num_examples 40 \
    --batch_size 1 \
    --dataset $DATA \
	--node_threshold $NODE \
	--edge_threshold $EDGE \
	--aggregation $AGG \
    --example_length 64 \
    --dict_id 10 \
	--nopair

# --submodules model.gpt_neox.layers.{}.attention.dense, model.gpt_neox.layers.{}.mlp.dense_4h_to_h,model.gpt_neox.layers.{}
