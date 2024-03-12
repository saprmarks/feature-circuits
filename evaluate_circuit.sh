#!/bin/bash

CIRCUIT_PATH=$1
DATA=$2

python evaluate.py \
    $CIRCUIT_PATH \
    --model EleutherAI/pythia-70m-deduped \
    --dataset /share/projects/dictionary_circuits/data/phenomena/${DATA}.json \
    --num_examples 150 \
	--dict_id id
