## Installation

Python version 3.10 (this is what works for @canrager, which py version is intended?)

Init submodule using 
```
git submodule init
git submodule update
```

## Shared Data
Sam: For any data that's being shared across everyone on the project, we store that in `/share/projects/dictionary_circuits`. There you'll see the state_dicts for the autoencoders and linguistic data for doing patching experiments.
I usually use the autoencoders from the 1_32768 series (there's one for each layer's MLP). See the readme here [https://github.com/saprmarks/dictionary_learning] for more info on these autoencoders (the open-source dictionaries mentioned in the readme are just the ones stored in `/share/projects/dictionary_circuits/autoencoders` (so please don't overwrite them!!)).
Aaron also trained autoencoders for other model components, e.g. attention heads, which you can find in the same place.
the folder with the autoencoder has a config.json file with the hyperparameters used to train it.


## Circuit discovery:
- `attribution.py`: Functions for estimating the effect of corrupting the activation of dictioary features *on a single metric*
- `acdc.py`: Wrappers for applying attribution functions to (1) a loss metric applied to logits and (2) on downstream dictionary features.
- `Circuit.locate_circuit()` in `circuit.py`: Iteratively measuring direct and indirect effect on a loss metric, to map out the feature circuit.
- `./notebooks/plot_circuit.py`: Visualize the feature circuit


## Feature Clustering
- `quanta-discovery/misc/create_pile_canonical.py`: Tokenize a chunk of The Pile (600k documents currently)
- `quanta-discovery/scripts`: Evaluate cross entropy loss for predicting the next token `y` given the sequence `context` – for every document in the dataset created above
- `./feature_clustering/feature_grad_cache.py`: Save feature activations and gradients for `contexts` with low loss
    - NNsight doc: https://nnsight.net/
- `./feature_clustering/cat_results.py`: Merge results in the many files created by `feature_grad_cache.py` into one big file.
- `./feature_clustering/clustering_svd.py`: Apply spectral clustering to the saved feature activations: (1) `Activations` (2) `Activations * Gradients`. Optionally reducing dimensionality with SVD.

- Streamlit webapp: https://github.com/canrager/feature-clustering-webapp. Separate repo which is directly connected to the streamlit hosting service. Running streamlit locally is great for debugging.
