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