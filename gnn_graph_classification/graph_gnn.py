import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool


class GraphGNN(nn.Module):
    def __init__(self, in_features, num_classes, gcn_dim=[16], dropout=0.5):
        super().__init__()
        torch.manual_seed(42)
        gnn_layers = [GraphConv(in_features, gcn_dim[0])]  # note that we applied GraphConv instead GCNConv here, it
        # ignored neighborhood normalization and add skip connection to the central node info
        for i in range(1, len(gcn_dim)):
            gnn_layers.append(GraphConv(gcn_dim[i - 1], gcn_dim[i]))
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.linear = Linear(gcn_dim[-1], num_classes)
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
            x = self.relu_layer(x)
            x = self.dropout_layer(x)
        x = global_mean_pool(x, batch)  # simply average the node embeddings to graph embedding
        x = self.relu_layer(x)
        x = x = self.dropout_layer(x)
        x = self.linear(x)
        return x