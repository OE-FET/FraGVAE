# Fragment Graphical Variational AutoEncoder
## Overview

This repository contains the framework and code for constructing a Fragment Graphical Variational AutoEncoder (FraGVAE) for use with molecular graphs, as described in our published and [arXiv](https://arxiv.org/abs/1910.13325) articles. As directly decoding large graphs is [intractable](https://arxiv.org/abs/1802.03480), here, we demonstrate the first fragment based autoencoder which directly autoencodes a bag of small graphs (subgraphs or fragments) which can be recombined to recreate the larger graph. Encoding is achieved through message passing neural networks ([MPNNs](https://arxiv.org/abs/1704.01212)). FraGVAE appears promising for generating compressed representations of small datasets because the algorithm is designed such that the number of ways to reconstruct the molecular graph typically scales factorially with the size of the molecular graph.  

To demonstrate the FraGVAE process below we provide the flowchart of the autoencoding of the small molecule Ibruprofen. First FraGVAE encodes the fragment bag and connectivity of the fragments using MPNNs to generate the latent space [Zf, Zc]. After the bag of fragments are autoencoded, reconstruction initializes by randomly selecting a starting fragment from the fragment bag to nucleate reconstruction of the large molecular graph. The decoding algorithm ranks every proposed graph creating a decision tree-based flow of a molecular graph generation. All valid subgraphs are marked with red arrows, graph ranks are marked with dashed arrows and paths selected are marked with solid red arrows.

<p align="center">
    <img width="" height="" src="https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/FraGVAE_flowchart_Ibuprofen.png">
</p>


### Encoding graphs

The encoding networks are convolutional message passing network based on Duvenaud et. al. [convolutional networks](https://arxiv.org/pdf/1509.09292.pdf) to generated ECFP. Here the network is modified such that feature vectors corresponding to the neighboring atoms and bonds [A<sub>w</sub> ,B<sub>w,v</sub>] for each atom (v) are sent though a neural network before being pooled (summed) at each atom. This additional step removes inherent bond feature exchange symmetry in directly pooling neighboring atom and bond features. In addition at each iteration (i), the graphical encoder updates B<sub>i,w,v</sub> enabling the network to control the flow of certain information through the graphical network as well as generating circular fingerprints centered on bonds. The neural message passing architecture for updating the atoms and bonds are seen in the two equations below:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=A_{i&plus;1,v}&space;=&space;\phi(M_{i,&space;D(v)}[A_{i,v},&space;\sum_{w\epsilon&space;N(v)}\phi(M'_{i}[A_{i,w},B_{i,w,v}])]&plus;b_{i,&space;D(v)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A_{i&plus;1,v}&space;=&space;\phi(M_{i,&space;D(v)}[A_{i,v},&space;\sum_{w\epsilon&space;N(v)}\phi(M'_{i}[A_{i,w},B_{i,w,v}])]&plus;b_{i,&space;D(v)}))" title="A_{i+1,v} = \phi(M_{i, D(v)}[A_{i,v}, \sum_{w\epsilon N(v)}\phi(M'_{i}[A_{i,w},B_{i,w,v}])]+b_{i, D(v)}))" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=B_{i&plus;1,w,v}&space;=&space;\phi(M_{i}[B_{i,u,v},A_{i,u}&plus;A_{i,v}])]&plus;b_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B_{i&plus;1,w,v}&space;=&space;\phi(M_{i}[B_{i,u,v},A_{i,u}&plus;A_{i,v}])]&plus;b_{i})" title="B_{i+1,w,v} = \phi(M_{i}[B_{i,u,v},A_{i,u}+A_{i,v}])]+b_{i})" /></a>
</p>

where N(v) is the set of neighboring elements of v, D(v) is the degree of vertex v (number of edges connected to v), phi is an arbitary activation function, M and b correspond to the neural network weight matrices and basies vectors for atoms, bonds and message neural networks. The latent space [Zf, Zc] is created identically to Duvenaud et. al. with a series of sparsifying neural networks used to map the updated atom and bond features to vector space which are then pooled to create both [Zf, Zc].  Zc contains two neural passing networks, as the secondary orthogonal F' network is used to generate fingerprints for completed and partial rings. In F' only atoms and edges (including any aromatic fragment) present in rings are included, the bond features are not updated.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Z_{F}&space;=&space;\sum_{v}\phi&space;(MA_{i,v}&plus;b)&space;=&space;\sum_{v}Z_{F,v}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z_{F}&space;=&space;\sum_{v}\phi&space;(MA_{i,v}&plus;b)&space;=&space;\sum_{v}Z_{F,v}" title="Z_{F} = \sum_{v}\phi (MA_{i,v}+b) = \sum_{v}Z_{F,v}" /></a>
</p>
<p align="center">
 <a href="https://www.codecogs.com/eqnedit.php?latex=Z_{C}&space;=&space;\sum_{i=1...n,v}(\phi(M_{i}A_{i,v}&plus;a_{i})&space;&plus;\phi(M'_{i}A'_{v,i}&plus;b_{i})&plus;\sum_{u\epsilon&space;N(v)&space;}\phi(M''_{i}B_{i,u,v}&plus;c_{i})&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z_{C}&space;=&space;\sum_{i=1...n,v}(\phi(M_{i}A_{i,v}&plus;a_{i})&space;&plus;\phi(M'_{i}A'_{v,i}&plus;b_{i})&plus;\sum_{u\epsilon&space;N(v)&space;}\phi(M''_{i}B_{i,u,v}&plus;c_{i})&space;)" title="Z_{C} = \sum_{i=1...n,v}(\phi(M_{i}A_{i,v}+a_{i}) +\phi(M'_{i}A'_{v,i}+b_{i})+\sum_{u\epsilon N(v) }\phi(M''_{i}B_{i,u,v}+c_{i}) )" /></a>
 </p>


where Z<sub>F,v</sub> if the encoded representation of the fragment centered on atom v, the initial states of the bond and atoms features are A<sub>0,v</sub>, and B<sub>0,u,v</sub> which represent the molecular graph and n is the number of graph convolutions. A variational layer is added into each sparsify network (F), in an identical architecture to [Gómez-Bombarelli](https://pubs.acs.org/doi/full/10.1021/acscentsci.7b00572). 


### Decoding bag of fragments


The specifics of the decoding of the bag and connectivity of fragments are demonstrated below in two GIFs. The fragment decoder decodes a bag of circular [fragments](https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/ECFP_example.png) centered on an atom (green), similar to extended connectivity fingerprints ([ECFP](https://pubs.acs.org/doi/pdf/10.1021/ci100050t)), with radius one (standard terminology used in chemistry). These small circular fragments are connected to 'dangling atoms' (cyan) which signify the nearest neighboring node connected via a specific bond. The bag of fragments is decoded by iteratively sampling the fragment decoder to determine the probability of the specific center node and nearest neighbor nodes given Zf, previously selected circular fragments, centered nodes and nearest neighbors. The GIF is generated in F1_model.py. **Note Chrome does not support the starting/stopping of GIFs. To walk through the GIFs please install a chrome add-on such as [GIF Scrubber](https://chrome.google.com/webstore/detail/gif-scrubber/gbdacbnhlfdlllckelpdkgeklfjfgcmp?hl=en) or click on the GIF to go to YouTube**.

[![GIF of recombination of fragments](https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/Ibuprofen_F1.gif)](https://www.youtube.com/watch?v=fiykijkK9ls)


Sampling a fragment occurs in two stages, the first stage samples the center node (n<sub>j</sub>) of the fragment and the second is an iterative process that samples the nearest neighbors (nn<sub>j</sub><sup>d</sup>) of the fragment given previous decisions. The sampling stops when the terminating node and nearest neighbors node are sampled. P(n<sub>j</sub>)) and P(nn<sub>j</sub>)) can be described below:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P(n_{j})&space;=&space;F_{n}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j}&space;,\sum_{j'=0}^{j-1}Z_{f,j}&space;])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(n_{j})&space;=&space;F_{n}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j}&space;,\sum_{j'=0}^{j-1}Z_{f,j}&space;])" title="P(n_{j}) = F_{n}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j} ,\sum_{j'=0}^{j-1}Z_{f,j} ])" /></a>
 </p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P(nn_{j}^{d})&space;=&space;F_{nn}^{d}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j}&space;,\sum_{j'=0}^{j-1}Z_{f,j},&space;n_{j},&space;\sum_{d'=0}^{d-1}nn_{j}^{d'}&space;])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(nn_{j}^{d})&space;=&space;F_{nn}^{d}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j}&space;,\sum_{j'=0}^{j-1}Z_{f,j},&space;n_{j},&space;\sum_{d'=0}^{d-1}nn_{j}^{d'}&space;])" title="P(nn_{j}^{d}) = F_{nn}^{d}([Z_{f}-\sum_{j'=0}^{j-1}Z_{f,j} ,\sum_{j'=0}^{j-1}Z_{f,j}, n_{j}, \sum_{d'=0}^{d-1}nn_{j}^{d'} ])" /></a>
  </p>
  
Where F<sub>n</sub> and F<sub>nn</sub> are deep neural networks. Training occurs by randomly sampling every fragment in a molecular structure by and minimizing the cross entropy loss of the P(n<sub>j</sub>)) and P(nn<sub>j</sub>)) compared to the ground truth. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}(f_{rags})&space;=&space;\sum_{j}&space;D_{KL}(P(n_{j})||\widehat{P}(n_{j}))&space;&plus;\sum_{j,d}&space;D_{KL}(P(nn_{j}^{d})||\widehat{P}(nn_{j}^{d}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(f_{rags})&space;=&space;\sum_{j}&space;D_{KL}(P(n_{j})||\widehat{P}(n_{j}))&space;&plus;\sum_{j,d}&space;D_{KL}(P(nn_{j}^{d})||\widehat{P}(nn_{j}^{d}))" title="\mathcal{L}(f_{rags}) = \sum_{j} D_{KL}(P(n_{j})||\widehat{P}(n_{j})) +\sum_{j,d} D_{KL}(P(nn_{j}^{d})||\widehat{P}(nn_{j}^{d}))" /></a>
 </p>

### Decoding complete molecule

All the possible larger fragments are created through the combination of small circular fragments with radius one. Please note larger circular fragments centered on an atom or bond are defined as fragments which include all bonds/atoms within a fixed radius centered on an atom/bond respectively. Construction of larger fragments initiates by selecting a random fragment to nucleate the construction of the graph. Using a decision tree process, a route through the tree is selected by the decoder ranking all proposed fragments using the connectivity Zc of the current, previous and encoded fragments as inputs. All proposed fragments are generated by adding every valid connection between two dangling bonds on the emerging molecule and adding every possible fragment in the fragment bag to every dangling bond in the emerging graph. This process iteratively removes dangling bonds (currently solving dark blue bond) to generate a larger molecular graph. This process terminates when all dangling bonds are accounted for. The GIF is generated in FHO_model.py. A GIF of the process to reconstruct the original molecular graph is demonstrated below.

[![GIF of recombination of fragments](https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/Ibuprofen_FHO.gif)](https://www.youtube.com/watch?v=b-27VvGA6R8)

The graph with the highest rank (G<sub>j*</sub><sup>i</sup> where G<sub>j*</sub><sup>i</sup>∈G<sub>J</sub><sup>i</sup> and (G<sub>J</sub><sup>i</sup>={G<sub>0</sub><sup>i</sup>,…,G<sub>j</sub><sup>i</sup>})) is selected as the next appropriate subgraph step to rebuild G. The set of G<sub>J</sub><sup>i</sup> includes all valid graphs where a sampled fragment can connect to a dangling bond in G<sub>j*</sub><sup>i-1</sup> as well as any possible ring combination that can be formed by connecting any two dangling bonds in G<sub>j*</sub><sup>i-1</sup>. The graph ranking is specifically determined by estimating the z-score, a measure of model confidence, between the invalid and valid classes. Deep learning is used to model the P(G<sub>j</sub><sup>i</sup> |Zc, G<sub>j*</sub><sup>i-1</sup>).

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P(G_{j}^{i}|Z_{c},G_{j*}^{i-1})&space;=&space;F(E_{Zc}(G),E_{Zc}(G_{j*}^{i-1}),E_{Zc}(G_{c}{j}^{i}),T(G_{c}{j}^{i},G_{j*}^{i-1},G))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(G_{j}^{i}|Z_{c},G_{j*}^{i-1})&space;=&space;F(E_{Zc}(G),E_{Zc}(G_{j*}^{i-1}),E_{Zc}(G_{c}{j}^{i}),T(G_{c}{j}^{i},G_{j*}^{i-1},G))" title="P(G_{j}^{i}|Z_{c},G_{j*}^{i-1}) = F(E_{Zc}(G),E_{Zc}(G_{j*}^{i-1}),E_{Zc}(G_{c}{j}^{i}),T(G_{c}{j}^{i},G_{j*}^{i-1},G))" /></a>
 </p>

Where E<sub>Zc</sub> is a function encoding [Zc] for any graph input G' and T calculates the Tanimoto index between graphs. The training error for a specific training example is minimizing the cross entropy loss of the P(G<sub>j</sub><sup>i</sup> |Zc, G<sub>j*</sub><sup>i-1</sup>) compared to the ground truth. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{L}(f_{rags})&space;=&space;\sum_{j,i}&space;D_{KL}(P(G_{j}^{i}|Z_{c},G_{j*}^{i-1})&space;||\widehat{P}(G_{j}^{i}|Z_{c},G_{j*}^{i-1}&space;))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{L}(f_{rags})&space;=&space;\sum_{j,i}&space;D_{KL}(P(G_{j}^{i}|Z_{c},G_{j*}^{i-1})&space;||\widehat{P}(G_{j}^{i}|Z_{c},G_{j*}^{i-1}&space;))" title="\mathcal{L}(f_{rags}) = \sum_{j,i} D_{KL}(P(G_{j}^{i}|Z_{c},G_{j*}^{i-1}) ||\widehat{P}(G_{j}^{i}|Z_{c},G_{j*}^{i-1} ))" /></a>
 </p>

### Short term updates:
- [ ] New user issues (please suggest additions, package modifications)
- [ ] Add option to include charges on atoms
- [ ] Add option to include label for single/double/triple bonds as in rings when decoding fragments
- [ ] Automatic Bayesian optimization of molecules
- [ ] Distributed hyperparameter tuning (integration into Spark and or AWS)
- [ ] More tutorials


### Long term updates:
- [ ] Include Tesnsorflow GPU Options
- [ ] Replacing atoms with clusters
- [ ] Convert to Tensorflow 2.0

## Training Data Generation

As it is impossible to marginalize over every possible graph [reconstruction](https://arxiv.org/abs/1805.09076), here we use generators to train our models by randomly selecting a reconstruction pathway through the graph. As this process is non-trivial, we visualize the process below in two animations for expedited troubleshooting. For specific formal details see the [supplementary information]().


### Fragment Training Data generation

Fragment autoencoding training occurs by minimizing the Kullback–Leibler divergence between the probability of the model selecting a fragment subcomponent and the ground truth. The specific fragment, center node and near neighbor are randomly selected (see below). To create similar animations please use F1_test_utils.py (requires updating). 
[![F1_training data](https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/F1_test_utils_gif_02.gif)](https://www.youtube.com/watch?v=ywRup_eu__I)

### Fragment Connectivity Training Data generation

Fragment autoencoding training occurs by minimizing the Kullback–Leibler divergence between the probability of the model selecting an emerging molecular structure and the ground truth if the proposed graph is a sub-graph of the final molecular graph. Appropriate and inappropriate emerging graphs are marked in green and red respectively. The generator randomly attempts to form rings by connecting two random dangling bonds and attaching valid fragments from the fragment bag to dangling bond in the emerging graph. Please use the gif generator functionality in FHO generators.py to generate similar GIFs.

[![F1_training data](https://github.com/OE-FET/FraGVAE/blob/master/imgs_gifs/FHO_generator_gif01.gif)](https://www.youtube.com/watch?v=wFQ_lzSpu6Y)

### Questions, problems, collaborations?
For questions or problems please create an [issue](https://github.com/OE-FET/FraGVAE/issues). Please raise issues if sections of code require further explanation. For any collaboration ideas for fundamental concept modifications and extensions please contact me directly (jwarmitage@gmail.com). Always happy to chat. :D

## How to install
### Requirements
Please check [environment.yml](https://github.com/OE-FET/FraGVAE/blob/master/environment.yml) file. Core packages include:
- numpy
- rdkit
- pandas
- keyboard
- tensorflow == 1.14
- cairosvg == 2
- matplotlib == 3

Please create issue if there is a problem.

### Pip Installiation

pip install git+https://github.com/OE-FET/FragVAE

## Components

### Example files
- **Introduction_to_FraGVAE.ipynb** - A jupyter notebook file demonstrating how to generate, train and test simple functionality of FraGVAE.
- **Run_characterization_example.ipynb** - A jupyter notebook file demonstrating how to run testing seen in publication based on trained models. In our example, we perform encoding/decoding with the [ZINC](https://github.com/OE-FET/FraGVAE/tree/master/models/experiment000001), [ESOL](https://github.com/OE-FET/FraGVAE/tree/master/models/experiment000002) and our [organic additive](https://github.com/OE-FET/FraGVAE/tree/master/models/experiment000004) dataset to generate a latent space to predict specific attributes in the small data regime. For all models the experiment parameter file (**exp.json**) contains all information regarding hyperparameters, training information and datasets.
- **characterization.py** - Support file with functions for characterizing datasets using FraGVAE

### FraGVAE Components
- **fragvae.py** - Main module containing FraGVAE object. Object controls training, saving, loading and testing all FragVAE models.
- **f1.py** -  Models for encoding and decoding ECFP fragments with a radius of 1 centered on atoms.
- **fho_model.py** - Models for encoding and decoding the connectivity of ECFP fragments with a radius of 1 centered on atoms using higher order fragments (ECFP with radius >1).
- **layers.py** - Custom graph Keras layers for F1 and FHO models
- **generators.py** - Tensorflow training data generators for F1 and FHO models
- **load_model_parameters.py** - defualt model hyperparameters initializer
- **display_funs.py** - Plotting and GIF creating functions
- **convert_mol_smile_tensor.py** - Functions for converting between smiles, RDKIT mols and tensor graphs
- **vae_callbacks.py** - Tensorflow training callbacks
- **filter_training_data.py** - functions for filtering training data to insure FraGVAE is stable
- **gen_paper_plots.py** - functions to generate plots for FraGVAE paper
- **utils.py** - Extra common functionality
- **f1_test_utils.py** - Testing functionality for F1 decoder. Compares true probabilities to predicted probabilities (requires updating).

## Authors:
The software is written by [John Armitage](https://github.com/jwarmitage) (jwarmitage@gmail.com) and [Leszek Spalek](https://github.com/LeszkoS). 

## Funding acknowledgements

This work was supported by the Canadian Centennial Scholarship Fund, Christ's College Cambridge and FlexEnable
