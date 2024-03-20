#!/bin/bash

# NODE=$1
# EDGE=$2
# AGG=$3

python circuit_clusters.py \
    --cluster_param_string lin_effects_final-5-pos_nsamples8192_nctx64 \
    --clusters_path /home/can/feature_clustering/app_clusters/ \
    --n_total_clusters 750 \
    --start_at_cluster 552 \
    --dict_size 32768 \
    --dict_path /share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/ \
    --dict_id 10 \
    --device cuda:0 \
    --model_name EleutherAI/pythia-70m-deduped \
    --batch_size 4 \
    --node_threshold 0.01 \
    --edge_threshold 0.1 \
    --aggregation sum \
    --samples_path /home/can/feature_clustering/app_contexts/samples8192.json \