import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from naive_gcn.naive_gcn_model import NaiveGCN
from torch_geometric.datasets import Planetoid


if __name__ == "__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    download_path = os.path.join(parent_directory, 'data')

    """
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
    """
    dataset = Planetoid(root=download_path, name='Cora')
    print(dataset[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    in_features = dataset.num_node_features
    num_classes = dataset.num_classes
    model = NaiveGCN(in_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')



