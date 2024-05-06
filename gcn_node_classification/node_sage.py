import copy
import torch
import torch.nn as nn
from torch_geometric.nn import (
    Aggregation,
    MaxAggregation,
    MeanAggregation,
    MultiAggregation,
    SAGEConv,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


class NodeSage(torch.nn.Module):
    def __init__(self, in_features, num_classes, gcn_dim=[16], dropout=0.5, aggr='mean', aggr_kwargs=None):
        """
        graph convolutional network
        :param in_features: the input feature dimension
        :param num_classes: the number of classes
        :param gcn_dim: the stack of gcn layers, the total stack layers equals 2 + len(gcn_dim) since we added a top
        and bottom layer to the stack
        :param aggr: the aggregation function, can be a string, aggregation module or a list of strings/aggregation
        modules, when appling a list of aggregation modules, t specifies multiple aggregation methods to use.
        The SAGEConv class will perform each of the specified aggregations on the neighboring features and concatenate
        the results.
        :param dropout: the dropout rate
        """
        super().__init__()
        torch.manual_seed(42)
        gcn_layers = [SAGEConv(in_features, gcn_dim[0], aggr=aggr, aggr_kwargs=aggr_kwargs)]
        for i in range(1, len(gcn_dim)):
            gcn_layers.append(SAGEConv(gcn_dim[i - 1], gcn_dim[i], aggr=aggr, aggr_kwargs=aggr_kwargs))
        gcn_layers.append(SAGEConv(gcn_dim[-1], num_classes, aggr=copy.deepcopy(aggr), aggr_kwargs=aggr_kwargs))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.relu_layer = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.gcn_layers) - 1):
            x = self.gcn_layers[i](x, edge_index)
            x = self.relu_layer(x)
            x = self.dropout_layer(x)
        x = self.gcn_layers[-1](x, edge_index)
        return x