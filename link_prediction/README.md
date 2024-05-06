# Graph Classification with Heterogeneous Graph Neural Networks

This module uses Heterogeneous Graph Neural Networks for link prediction tasks.

## Data

The data used in this project is a heterogeneous graph where nodes represent users and movies, and edges represent ratings given by users to movies. Each user and movie node has an embedding as a feature, besides each movie node also has a genre vector as a feature.

## Model

The model used in this project is a Heterogeneous Graph Neural Network model. It consists of two GraphSAGE convolution layers for updating node embeddings based on their neighborhood. The model is trained to predict the existence of edges between nodes, which is a link prediction task.
Note that this is a simple case that only includes one type of edge, for cases with multiple edge types, the model would create one instance for each edge type depending on the data.metadata().
## Usage

To run the code, follow these steps:

1. Follow the setup in the main Readme file
2. Run the `train_link_prediction.py` script to evaluate the model: `python train_link_prediction.py`
