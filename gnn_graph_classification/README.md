# Graph classification on citation networks

## Data Introduction

The MUTAG dataset is a collection of 188 chemical compounds, each represented as a graph, with nodes representing atoms and edges representing chemical bonds. This dataset is commonly used in graph machine learning tasks, particularly for graph classification problems.

Each graph in the MUTAG dataset is labeled with a binary class, indicating whether the chemical compound has a mutagenic effect on a bacterium or not. The dataset is thus used to train machine learning models to predict the mutagenicity of unseen chemical compounds based on their graph structure.

Here is the basic info about the dataset:
- Number of graphs: 188
- Number of features: 7
- Number of classes: 2

Looking into the first graph in the dataset, we can see 
Data(edge_index=[2, 38], x=[17, 7], edge_attr=[38, 4], y=[1])
indicating that the graph has 17 nodes, 38 edges, 7 features per node, 4 features per edge, and has one graph label.


## Model Introduction
We are building a simple graph classification model using Graph Neural Networks (GNNs). The model architecture is similar with node classification model, but with the following two differences:
1. The model outputs a single prediction for the entire graph, instead of one prediction per node. So we added an aggregation layer to combine node embeddings into a graph embedding.
2. We applied the GraphConv layer to the graph level representation to make the model aware of the graph structure. (ignored neighborhood normalization and add skip connection to the central node info)
