# Node classification on citation networks

## Data Introduction

### Cora Dataset
The 'Cora' dataset is a popular benchmark dataset in the field of graph-based machine learning.
It's a citation network where nodes represent documents and edges represent citation links between documents.
Each document is described by a binary word vector indicating the absence or presence of the corresponding word
from the dictionary.

- data.x: This is a [2708, 1433] dimensional tensor where 2708 is the number of nodes (documents) in the graph. 
Each row represents a document and is a binary vector that indicates the presence of a word in the document. 1433 
refers to the number of unique words in the dictionary. Each document is represented by a binary vector of length 
1433, where 1 indicates the presence of a word and 0 indicates the absence of a word.
- data.edge_index: This is a [2, 10556] dimensional tensor where 10556 is the number of edges (citations). 
Each column in the tensor represents an edge. For example, if edge_index[:, i] = [src, dst], this means that src 
cites dst for the ith edge.
- data.y: This is a tensor that contains the class labels of each document. The 'Cora' dataset has 7 classes, 
representing different topics of the documents.
- data.train_mask: This is a Boolean tensor that indicates which nodes should be included in the training set.
- data.val_mask and data.test_mask: These are Boolean tensors that indicate which nodes should be included in the 
validation and test sets, respectively. 

### PubMed Dataset
The PubMed dataset is a citation network where nodes represent scientific papers and edges represent citation links between these papers. It's a popular benchmark dataset in the field of graph-based machine learning.  Here's a breakdown of the data attributes:  
- data.x: This is a [19717, 500] dimensional tensor where 19717 is the number of nodes (papers) in the graph. Each row represents a paper and is a vector that indicates the presence of a word in the paper. 500 refers to the number of unique words in the dictionary. Each paper is represented by a vector of length 500, where the value indicates the presence of a word.  
- data.edge_index: This is a [2, 88648] dimensional tensor where 88648 is the number of edges (citations). Each column in the tensor represents an edge. For example, if edge_index[:, i] = [src, dst], this means that src cites dst for the ith edge.  
- data.y: This is a tensor that contains the class labels of each paper. The PubMed dataset has 3 classes, representing different topics of the papers.  
- data.train_mask, data.val_mask, and data.test_mask: These are Boolean tensors that indicate which nodes (papers) should be included in the training, validation, and test sets, respectively

## Model Introduction
We are applying a two GCN layers network on the 'Cora' dataset. In particular, a GCNConv layer is a fundamental building block of Graph Convolutional Networks, a type of neural network
designed to work directly with graphs. The key idea behind GCNs is to generate a node's new feature
representation by aggregating the feature information from its neighboring nodes and itself.
Here's a brief explanation of how GCNConv works:
1. Neighborhood Aggregation: For each node, GCNConv aggregates(sum) the features of its neighbors.
This is based on the assumption that a node's features can be represented by the features of its neighbors.
2. Transformation: The aggregated features are then transformed by a learnable weight matrix, which is the
main component that GCNConv learns during training. This transformation is a linear operation
(a matrix multiplication).
Note that the GCNConv does not include any non-linear activation function (e.g., ReLU) after the transformation. Therefore we applied a ReLU activation function after the first GCNConv layer.

The above node classification model is coded in node_gcn.py



