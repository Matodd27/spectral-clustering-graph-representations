# Graph Construction and Representation Learning for Spectral Clustering

This repository contains the code and experimental material for my MMath project:

**Graph Construction and Representation Learning for Spectral Clustering**

The project investigates how the construction of the similarity graph affects spectral clustering performance, and how learned representations from VAEs and SimCLR can improve the quality of the graph used for clustering.

## Project overview

Spectral clustering does not operate directly on the original data points. Instead, it first constructs a similarity graph, forms a graph Laplacian, computes a spectral embedding, and then applies a clustering algorithm such as k-means.

This project studies two main questions:

1. How sensitive is spectral clustering to the choice of graph construction?
2. Can learned representation spaces produce better similarity graphs for spectral clustering?

The experiments compare classical and adaptive graph construction methods across raw feature spaces and learned representation spaces.

## Methods

The repository includes implementations and experiments for:

- k-nearest-neighbour Gaussian graphs
- fully connected Gaussian graphs
- Clustering with Adaptive Neighbours (CAN)
- Projected Clustering with Adaptive Neighbours (PCAN)
- biclique-kernel-based graph construction
- Variational Autoencoders (VAEs)
- SimCLR representations
- downstream spectral clustering evaluation

## Datasets

Experiments are run on a mixture of small benchmark datasets and image datasets, including:

- Iris
- Wine
- Seeds
- Breast Cancer Wisconsin
- MNIST
- FashionMNIST
- CIFAR-10

## Installation

Clone the repository:

    git clone https://github.com/<username>/spectral-clustering-graph-representations.git
    cd spectral-clustering-graph-representations

Install the required dependencies:

    pip install -r requirements.txt

If the project is structured as an editable Python package:

    pip install -e .

## Results

The main experimental results compare:

- graph construction methods in raw feature spaces;
- representation learning methods for downstream spectral clustering;
- graph construction methods inside learned representation spaces;
- clustering accuracy and computational cost.

The key finding is that graph construction matters, but representation quality is often the more decisive factor on high-dimensional image datasets.

## Notes

Large datasets, trained model checkpoints, and generated result files may not be included directly in this repository. For example, the MNIST, FashionMNIST and CIFAR-10 SimCLR and VAE latents are not included. However, these can be reproduced when necessary by following the training procedure defined in the corresponding files.

## Acknowledgements

This project draws on literature from spectral clustering, graph construction, adaptive neighbours, hypergraph-based clustering, variational autoencoders, and contrastive representation learning. With some code, particularly the MNIST and FashionMNIST VAEs and knn_graph functions inspired by Jeff Calder's GraphLearning repository.

https://github.com/jwcalder/GraphLearning 

## Licence

This repository is intended for academic use. See `LICENSE` for details.
