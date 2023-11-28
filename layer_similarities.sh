#!/bin/bash

submodules=""
autoencoders=""
for layer in {0..4}; do
    submodules+="model.gpt_neox.layers.${layer}.mlp.dense_4h_to_h,"
    autoencoders+="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer${layer}/0_8192/ae.pt,"
done
submodules+="model.gpt_neox.layers.5.mlp.dense_4h_to_h"
autoencoders+="/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer-1/3_8192/ae.pt"

python similarity_analysis.py \
    --models "EleutherAI/pythia-70m-deduped" \
    --submodules $submodules \
    --autoencoders $autoencoders \
    --num_examples 100