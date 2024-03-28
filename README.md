# Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

This repository contains code, data, and links to autoencoders for replicating the experiments of [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](TODO). 


## Installation

Use python >= 3.10. To install dependencies, use
```
pip install -r requirements.txt
```

You will also need to clone the [dictionary learning repository](https://github.com/saprmarks/dictionary_learning). Run this command from the root directory of this repository to get that code:
```
git submodule update --init
```


## Data
### Subject–Verb Agreement
We create modified versions of the stimuli from [Finlayson et al. (2021)](https://aclanthology.org/2021.acl-long.144/) (code [here](https://github.com/mattf1n/lm-intervention)). Specifically, we use the same nouns and structures, but modify the verb sets to only include those whose singular and plural inflections are single tokens in Pythia. Our data may be found in `data/`.

We use different splits for discovering circuits and evaluating faithfulness/completeness. The `*_train` files are those we use to discover circuits (typically with 100-example subsamples); the `*_test` files are those we use for evaluation.

### Bias in Bios
We download the Bias in Bios dataset from [Huggingface](https://huggingface.co/datasets/LabHC/bias_in_bios). We write functions to subsample the data for our classifier experiments; see [the Bias in Bios experiment notebook](experiments/bib_shift.ipynb).

### Cluster Data
We provide an online interface for observing and downloading clusters [here](https://feature-circuits.xyz).


## Autoencoders
To run our experiments, you will need to either train or download sparse autoencoders for each layer of Pythia 70M. See the [dictionary learning repository](https://github.com/saprmarks/dictionary_learning) for download links and training instructions.


## Experiments
Here, we provide instructions for replicating the results from our paper.

### Subject–Verb Agreement
To discover a circuit, use the following command:
```
scripts/run_circuit_triangles.sh <data_type> <node_threshold> <edge_threshold> <aggregation> <dict_id>
```
For example, to discover a sparse feature circuit for agreement across a relative clause using node threshold 0.1 and edge threshold 0.01, and with no aggregation across token positions, run this command:
```
scripts/run_circuit_triangles.sh rc_train 0.1 0.01 none 10
```
If you would like a circuit composed of model components instead of sparse features, replace "10" with "id".

By default, this will save a circuit in `circuits/`, and a circuit plot in `circuits/figures/`.

### Bias in Bios
All code for replicating our data processing, classifier training, and SHIFT method (including all baselines and skylines) can be found in [experiments/bib_shift.ipynb](experiments/bib_shift.ipynb).

To generate a circuit for the BiB classifier, use [experiments/bib_circuit.ipynb](experiments/bib_circuit.ipynb)

### Clusters
After downloading a cluster, run this script:
```
scripts/run_circuit_triangles_clusters.sh <data_path> <node_threshold> <edge_threshold> <dict_id>
```
`data_path` should be the full path to a cluster .json in the same format as those that can be downloaded [here](https://feature-circuits.xyz). By default, this will save a circuit in `circuits/` and a circuit plot in `circuits/figures/`.


## Citation
If you use any of the code or ideas presented here, please cite our paper:
```
@article{marks-etal-2024-feature,
    author={Samuel Marks and Can Rager and Eric J. Michaud and Yonatan Belinkov and David Bau and Aaron Mueller},
    year={2024},
    journal={Computing Research Repository},
    volume={arXiv:TODO},
    url={TODO}
}
```


## License
We release source code for this work under an MIT license.