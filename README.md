# Fragment Graphical Variational AutoEncoder
## Overview
This repository contains the framework and code for constructing a **Fra**gment **G**raphical **V**ariational **A**uto**E**ncoder (FraGVAE) for use with molecular graphs, as described in our [published]() and [preprint]() articles. This approach appears promosing for generating compressed circular fingerprints representation for small datasets as well a geneartive model for molecular graphs. 

FraGVAE autoencodes molecular graphs. A molecular graph is described by 3 tensors; atoms (graph nodes), edges (connecitivty of nodes) and bonds (graph edge labels). Encoding is achieved through [message passing neural networks](https://arxiv.org/abs/1704.01212) (MPNN). As directly decoding large graphs is [intractable](https://arxiv.org/abs/1802.03480), our work focuses on a novel fragment based decodeing aproach. As it is impossbile to directly decode large graphs first we decode a bag of small graphs (fragments) followed by recombining the small graphs to reconstruct the larger graph. To demonstrate the FraGVAE process below we provide the flowchart of the autoencoding of the small molecule Ibruprofen. 

<p align="center">
    <img width="" height="" src="https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/FraGVAE_flowchart_Ibuprofen.png">
</p>

The specifics of the decoding of the bag and connecitivty of fragments is demonstrated below in two GIFs. The fragment decoder decodes a bag of circular [fragments](https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/ECFP_example.png) centered on a atom (green) , similiar to extended connectivity fingerprints ([ECFP](https://pubs.acs.org/doi/pdf/10.1021/ci100050t)), with radius one. Each fragment is connected to 'dangling atoms' (cyan) which signify the nearest neighbouring node connected via a specific bond. The bag of fragments is decoded by itereatively sampling the fragment decoder to determine the probilility of the specific center node and nearest neighbour nodes given Zf, previous selected circular fragments, centered nodes and nearest neighbours. GIF is generated in **F1_model.py**. 

<p align="center">
    <img width="" height="" src="https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/Ibuprofen_F1.gif">
</p>

Seen below using higher order fragment information the bag of fragments are recombined to reconsutruct the orgnial molecular graph. Reconstruction initiates by selelcting a random fragment to nucleate the reconstruction of the graph. Iteratively, all possible fragments from the bag of fragments are added and ranked to all appropriate dangling bonds in the emerging graph (brown bonds/atoms). This process iteratively removes dangling bonds (currently solvening dark blue bond) to generate a larger molecular graph given the information in Zc. This process terminates when all dangling bonds are accounted for. GIF is generated in **FHO_model.py**. 

![GIF of recombination of fragments](https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/Ibuprofen_FHO.gif)

### Short term updates:
- [ ] New user issues (please suggest additions, package modifications) 
- [ ] Add option to include charges on atoms
- [ ] Add option to include label for single/double/triple bonds as in rings when decoding fragments
- [ ] Automatic bayesian optimziation of molecules
- [ ] More tutorials


### Long term updates:
- [ ] Include Tesnsorflow GPU Options
- [ ] Replacing atoms with clusters
- [ ] Convert to Tensorflow 2.0

## Training Data Generation
As it is impossible to margninzale over every possible graph [reconstruction](https://arxiv.org/abs/1805.09076), here we use generators to train our models by randomly selecting a reconsctruction path way through the graph. As this process is non-trivial, we visualize the process below in two animations for expedite troubleshooting. For specific formal details see the [supplemnetary information](). 

### Fragment Training Data generation

Fragment autoencoding training occurs by minizing the Kullback–Leibler divergence between the probaiblity of the model selecting a fragment subcomponent and the ground turth. The specific fragment, center node and near neighbour are randomly selected (see below). To create similar animations please use **F1_test_utils.py** (requires updating).  
![F1_training data](https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/F1_test_utils_gif_02.gif)

### Fragment Connectivity Training Data generation

Fragment autoencoding training occurs by minizing the Kullback–Leibler divergence between probabily of the model selecting an emerging molecular structure and the ground turth if the proposed graph is a sub-graph of the final molecular graph. Appropriate and inappropriate emerging graphs are marked in green and red respectively. The generator randomly attempts to form rings by connecting two random dangiling bonds and attaching valid fragments from the fragment bag to dangling bond in the emerging graph. Please use the gif generator functionality in FHO **generators.py** to generate similar GIFs.

![F1_training data](https://github.com/LeszkoS/FraGVAE/blob/master/imgs_gifs/FHO_generator_gif01.gif)

### Questions, problems, collaborations?
For questions or problems please create an [issue](https://github.com/LeszkoS/FraGVAE/issues). Please raise issues if sections of code is requires further explinatiation. For any collaborition ideas for fundamental concept modificaitons and extensions please contact me direclty (jwarmitage@gmail.com). Always happy to chat. :D
## How to install

### Requirements
Please check [environment.yml](https://github.com/LeszkoS/FraGVAE/blob/master/environment.yml) file. Core packages include:
- numpy
- rdkit
- pandas
- keyboard
- tensorflow >= 1.13
- cairosvg == 2
- matplotlib == 3

### Pip Installiation

pip install git+https://github.com/OE-FET/FragVAE

## Components

### Example files
- **Introduction_to_FraGVAE.ipynb** - A jupyter notebook file demonstrating how to generate, train and test simple functionality of FraGVAE.
- **Training_FraGVAE_publication.ipynb** - A jupyter notebook file dmonstrating how to run testing seen in publication based on trained models. In our example, we perform encoding/decoding with the [ZINC](https://github.com/LeszkoS/FraGVAE/tree/master/models/experiment000001), [ESOL](https://github.com/LeszkoS/FraGVAE/tree/master/models/experiment000002) and our [organic additive](https://github.com/LeszkoS/FraGVAE/tree/master/models/experiment000004) dataset to generate a latent space to predict specific attributes in the small data regime. For all models the eperiment parameter file (**exp.json**) contains all information regarding hyperparameters, training information and datasets. 

### FraGVAE Components
- **fragvae.py** - Main module containing FraGVAE object. Object controls training, saving, loading and testing all FragVAE models.
- **F1_model.py** -  Models for encoding and decoding ECFP fragments with a radius of 1 centered on atoms. 
- **FHO_model.py** - Models for encoding and deconding the connectivity of ECFP fragments with a radius of 1 centered on atoms using higher order fragments (ECFP with radius >1). 
- **layers.py** - Cutsom graph Keras layers for F1 and FHO models
- **generators.py** - Tensorflow training data generators for F1 and FHO models
- **characterization.py** - Functions for characterizing datasets using FraGVAE 
- **load_model_parameters.py** - defualt model hyperarameters initlizer
- **display_funs.py** - Plotting and GIF creating functions
- **convert_mol_smile_tensor.py** - Functions for converting between smiles, RDKIT mols and tesnor graphs
- **vae_callbacks.py** - Tensorflow training callbacks
- **filter_training_data.py** - functions for filtering training data to insure FraGVAE is stable
- **gen_paper_plots.py** - functions to generate plots for FraGVAE paper
- **utils.py** - Extra common functionality
- **F1_test_utils.py** - Testing functionalty for F1 decoder. Compares true probabilities to predictied probabilities (requires updating).


## Authors:
The software is written by [John Armitage](https://github.com/jwarmitage) and [Leszek Spalek](https://github.com/LeszkoS). 

## Funding acknowledgements

This work was supported by the Canadian Centennial Scholarship Fund, Christ's College Cambridge and FlexEnable
