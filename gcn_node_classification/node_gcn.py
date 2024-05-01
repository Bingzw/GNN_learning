import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class NodeGCN(torch.nn.Module):
    def __init__(self, in_features, num_classes, gcn_dim=[16], dropout=0.5):
        """
        graph convolutional network
        :param in_features: the input feature dimension
        :param num_classes: the number of classes
        :param gcn_dim: the stack of gcn layers, the total stack layers equals 2 + len(gcn_dim) since we added a top
        and bottom layer to the stack
        :param dropout: the dropout rate
        """
        super().__init__()
        torch.manual_seed(42)
        gcn_layers = [GCNConv(in_features, gcn_dim[0])]
        # away information, so we need multiple layers to aggregate multi-steps information
        for i in range(1, len(gcn_dim)):
            gcn_layers.append(GCNConv(gcn_dim[i - 1], gcn_dim[i]))
        gcn_layers.append(GCNConv(gcn_dim[-1], num_classes))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.gcn_layers) - 1):
            x = self.gcn_layers[i](x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=0.5, train=self.training)
        x = self.gcn_layers[-1](x, edge_index)
        return x