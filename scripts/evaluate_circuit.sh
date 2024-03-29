#!/bin/bash

# Run the ablation.py script with the specified arguments
python ablation.py \
--dict_path dictionaries \
--circuit rc_dict10_node0.1_edge0.01_n100_aggnone.pt \
--data rc_test.json \
--num_examples 30 \
--length 6 \
# arguments below are optional -- you probably don't need to change them
--dict_id 10 \
--dict_size 32768 \
--threshold 0.1 \
--ablation mean \
--handle_resids 'default' \
--start_layer -1 \
--device cuda:0

# --dict_path: Path to where you store your dictionaries
# --circuit: Filename in circuits folder
# --data: Filename in data folder
# -n: Number of samples
# -l: Length of inputs from test set. Omit for circuits of non-position-aligned features. Use 2 for simple, 5 for nounpp and within_rc, and 6 for rc
# --dict_id: Pass id for neuron circuits
# --dict_size: Size of dictionary
# --threshold: Node threshold
# --ablation: Alternatives: resample, zero
# --handle_resids: Alternatives: 'keep' and 'remove'; for deciding how to treat SAE error nodes
# --start_layer: Which layer to evaluate the circuit from; -1 means full circuit from embeddings
# --device: Device to run on