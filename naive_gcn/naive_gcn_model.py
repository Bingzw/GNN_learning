import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NaiveGCN(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        """
        A GCNConv layer is a fundamental building block of Graph Convolutional Networks, a type of neural network
        designed to work directly with graphs. The key idea behind GCNs is to generate a node's new feature
        representation by aggregating the feature information from its neighboring nodes and itself.
        Here's a brief explanation of how GCNConv works:
        1. Neighborhood Aggregation: For each node, GCNConv aggregates(sum) the features of its neighbors.
        This is based on the assumption that a node's features can be represented by the features of its neighbors.
        2. Transformation: The aggregated features are then transformed by a learnable weight matrix, which is the
        main component that GCNConv learns during training. This transformation is a linear operation
        (a matrix multiplication).
        :param dataset: the input dataset
        """
        super().__init__()
        self.conv1 = GCNConv(in_features, 16)  # note that a single layer only aggregate the 1 step away information,
        # so we need multiple layers to aggregate multi-steps information
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)