#!/bin/bash

python compute_similarity_matrix.py \
    --batch_size 1024 \
    --aggregation "None" \
    --results_dir "/home/can/dictionary-circuits/feature_clustering/clusters/dataset_nsamples32768_nctx16_tloss0.1_filtered-induction_attn-mlp-resid_pythia-70m-deduped" \
    --dict_path "/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped" \
    --d_model 512 \
    --dict_id 10 \
    --device "cuda:0"
#   --results_dir "/home/can/feature_clustering/clustering_pythia-70m-deduped_tloss0.1_nsamples128_npos32_filtered-induction_attn-mlp-resid" \
