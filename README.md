# Graph Neural Networks (GNN) Learning

This repository contains implementations of various Graph Neural Networks (GNNs) for node classification tasks.

## Table of Contents

- [Setup](#Setup)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)

## Setup

The project is implemented in Python and uses PyTorch and PyTorch Geometric. You can install the dependencies with pip:

1. clone the repo into your local machine
```
cd path/to/your/workspace
git clone https://github.com/Bingzw/DL_genAI_practicing.git
```
2. create python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
3. install the required packages
```
pip install -r requirements.txt
```
4. run model training
```
cd path/to/model/folder
python <train.py>
```

## Models & Tasks
The repository contains models solving the following tasks:
- Node classification: predicting the label of a node in a graph
  - GCN Model
  - GAT Model
  - GraphSAGE
- Graph classification: predicting the label of an entire graph
  - GNN based on GraphConv (this can also be changed to GCN, GAT or GraphSAGE)
- Link prediction: predicting whether an edge exists between two nodes
  - Heterogeneous GraphSAGE

## References
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- standford course: [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)

