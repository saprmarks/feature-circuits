# Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

This repository contains code, data, and links to autoencoders for replicating the experiments of [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647). 

## Demos and Links
- We provide an interface for exploring and downloading clusters [here](https://feature-circuits.xyz).
- View our SAEs and their features on [Neuronpedia](https://www.neuronpedia.org/p70d-sm).

## Installation

Use python >= 3.10. To install dependencies, use
```
pip install -r requirements.txt
```


## Data
### Subject–Verb Agreement
We create modified versions of the stimuli from [Finlayson et al. (2021)](https://aclanthology.org/2021.acl-long.144/) (code [here](https://github.com/mattf1n/lm-intervention)). Specifically, we use the same nouns and structures, but modify the verb sets to only include those whose singular and plural inflections are single tokens in Pythia. Our data may be found in `data/`.

We use different splits for discovering circuits and evaluating faithfulness/completeness. The `*_train` files are those we use to discover circuits (typically with 100-example subsamples); the `*_test` files are those we use for evaluation.

### Bias in Bios
We download the Bias in Bios dataset from [Huggingface](https://huggingface.co/datasets/LabHC/bias_in_bios). We write functions to subsample the data for our classifier experiments; see [the Bias in Bios experiment notebook](experiments/bib_shift.ipynb).

### Cluster Data
We provide an online interface for observing and downloading clusters [here](https://feature-circuits.xyz).

### Autoencoders
To run experiments with Pythia-70M, you will need to either train or download sparse autoencoders for each layer of Pythia 70M. You can download dictionaries by running this from the command line:
```
wget https://huggingface.co/saprmarks/pythia-70m-deduped-saes/resolve/main/dictionaries_pythia-70m-deduped_10.zip
unzip dictionaries_pythia-70m-deduped_10.zip
```
You should see the dictionaries in the `dictionaries/pythia-70m-deduped/` directory.

### Annotations
We provide feature annotations in `annotations/10_32768.jsonl`. These are primarily used in `circuit_plotting.py`.

## Experiments
Here, we provide instructions for replicating the results from our paper.

### Subject–Verb Agreement
To discover a circuit, use the following command:
```
scripts/get_circuit.sh <model> <data> <node_threshold> <edge_threshold> <aggregation>
```
For example, to discover a sparse feature circuit on Pythia-70m for agreement across a relative clause using node threshold 0.1 and edge threshold 0.01, and with no aggregation across token positions, run this command:
```
scripts/get_circuit.sh EleutherAI/pythia-70m-deduped rc_train 0.1 0.01 none
```
This script calls the main method of `circuit.py`, which is more flexible and can be used to run additional experiments (e.g. computing neuron circuits).

By default, this will save a circuit in `circuits/`, and a circuit plot in `circuits/figures/`.

To evaluate the **faithfulness** and **completeness** of circuits across a variety of thresholds, see [experiments/faithfulness.ipynb](experiments/faithfulness.ipynb). To evaluate just a single circuit, use the following command:
```
scripts/evaluate_circuit.sh <model> <circuit_path> <data> <node_threshold> <start_layer>
```
For example, to evaluate the faithfulness and completeness of the agreement across RC circuit starting at layer 2 with node threshold 0.1, you can run
```
scripts/evaluate_circuit.sh EleutherAI/pythia-70m-deduped circuits/pythia-70m-deduped_rc_train_n100_aggnone_node0.1.pt rc_test 0.1 2
```

### Bias in Bios
All code for replicating our data processing, classifier training, and SHIFT method (including all baselines and skylines) can be found in [experiments/bib_shift.ipynb](experiments/bib_shift.ipynb).

### Clusters
After downloading a cluster from [feature-circuits.xyz](https://feature-circuits.xyz), run this script:
```
scripts/get_circuit_nopair.sh <model> <data_path> <node_threshold> <edge_threshold>
```
`data_path` should be the full path to a cluster (without the `.json` suffix) in the same format as those that can be downloaded [here](https://feature-circuits.xyz). By default, this will save a circuit in `circuits/` and a circuit plot in `circuits/figures/`.


## General utilties
The following files contain utilities which are generally useful for our circuit discovery methods:
* `attribution.py` implements methods for attributing model behaviors to SAE features and error terms.
* `activation_utils.py` defines the `SparseAct` object, which bundles together feature activations and error terms in a convenient way and provides utilities for working with them in a unified way.
* `ablation.py` implements general methods useful for performing SAE feature ablations
* `circuit.py` contains our circuit discovery code
* `circuit_plotting.py` contains our code for plotting circuits, once discovered.
* `coo_utils.py` contains some utility functions for manipulating sparse-format tensors
* `dictionary_loading_utils.py` has helper functions for loading dictionaries
* `loading_utils.py` contains utilities for working with our subject-verb agreement datasets and clusters.

## Citation
If you use any of the code or ideas presented here, please cite our paper:
```
@inproceedings{marks2025sparse,
    title={Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models},
    author={Samuel Marks and Can Rager and Eric J Michaud and Yonatan Belinkov and David Bau and Aaron Mueller},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=I4e82CIDxv}
}
```


## License
We release source code for this work under an MIT license.
