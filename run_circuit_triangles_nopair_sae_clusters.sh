#!/bin/bash

NODE=$1
EDGE=$2
AGG=$3

python circuit_clusters.py \
    --cluster_param_string ERIC-QUANTA-CLUSTERS-ACTIVATIONS \
    --clusters_path /home/can/feature_clustering/app_clusters/ \
    --n_total_clusters 700 \
    --start_at_cluster 0 \
    --dict_size 32768 \
    --dict_path /share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/ \
    --dict_id 10 \
    --device cuda:0 \
    --model_name EleutherAI/pythia-70m-deduped \
    --batch_size 8 \
    --node_threshold $NODE \
    --edge_threshold $EDGE \
    --aggregation $AGG \
    --samples_path /home/can/feature_clustering/app_contexts/ERIC-QUANTA-CONTEXTS.json \