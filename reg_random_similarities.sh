#!/bin/bash

models="EleutherAI/pythia-70m-deduped,EleutherAI/pythia-70m-deduped,"
models+="random_models/EleutherAI/pythia-70m-deduped,"
models+="random_models/EleutherAI/pythia-70m-deduped-randomexceptembed"

autoencoders="autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768.pt,"
autoencoders+="autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768_seed14.pt,"
autoencoders+="autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768_randommodel.pt,"
autoencoders+="autoencoders/ae_mlp3_c4_lr0.0001_resample25000_dict32768_randomexceptembed.pt"

python similarity_analysis.py \
    --models "EleutherAI/pythia-70m-deduped" \
    --submodules "model.gpt_neox.layers.3.mlp.dense_4h_to_h" \
    --autoencoders $autoencoders \
    --plot_heatmap \
    --labels "Seed 1,Seed 2,Random,Random (except embeddings)" \
    --num_examples 5