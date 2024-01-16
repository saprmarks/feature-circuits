#!/bin/bash

python circuit.py \
    --model EleutherAI/pythia-70m-deduped \
    --submodules model.gpt_neox.layers.{}.mlp.dense_4h_to_h,model.gpt_neox.layers.{}.attention.dense \
    --num_examples 5 \
    --patch_method "separate" \
    --dataset /share/projects/dictionary_circuits/data/phenomena/rc.json

# --submodules model.gpt_neox.layers.{}.mlp.dense_4h_to_h,model.gpt_neox.layers.{}.attention.dense,model.gpt_neox.layers.{}