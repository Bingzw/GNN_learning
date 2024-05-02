import os
import torch
import torch.nn as nn
from tqdm import tqdm
from gcn_node_classification.node_gcn import NodeGCN
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def train_and_validate(data, model, optimizer, num_epochs):
    # In GNN, one graph is considered as a data point when doing batch stochastic gradient, since all nodes in the graph
    # somehow connected. However the dataset only contains one graph, so we don't need to worry about the batch in this
    # case.
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(data).argmax(dim=1)
            correct_pred = (prediction[data.val_mask] == data.y[data.val_mask]).sum()
            val_accuracy = int(correct_pred) / int(data.val_mask.sum())

    return model, val_accuracy


if __name__ == "__main__":
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    download_path = os.path.join(parent_directory, 'data')
    dataset = Planetoid(root=download_path, name='Cora', transform=NormalizeFeatures())
    print(dataset[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    in_features = dataset.num_node_features
    num_classes = dataset.num_classes

    learning_rate = [0.01, 0.001, 1e-4]
    weight_decay = [1e-4, 1e-5, 1e-6]
    gcn_dim = [[32], [16], [8]]
    dropout = [0.3, 0.5, 0.7]
    num_epochs = 200

    best_hyperparameters = {"best_model": None,
                            "best_lr": None,
                            "best_wd": None,
                            "best_dim": None,
                            "best_dp": None,
                            "best_val_acc": 0}
    for lr in learning_rate:
        for wd in weight_decay:
            for dim in gcn_dim:
                for dp in dropout:
                    print("training with lr: ", lr, " wd: ", wd, " dim: ", dim, " dp: ", dp)
                    model = NodeGCN(in_features, num_classes, dim, dp).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    model, val_acc = train_and_validate(data, model, optimizer, num_epochs)
                    print("validation accuracy: ", val_acc)
                    print("-----------------------------------")
                    if val_acc > best_hyperparameters["best_val_acc"]:
                        best_hyperparameters["best_val_acc"] = val_acc
                        best_hyperparameters["best_model"] = model
                        best_hyperparameters["best_lr"] = lr
                        best_hyperparameters["best_wd"] = wd
                        best_hyperparameters["best_dim"] = dim
                        best_hyperparameters["best_dp"] = dp
    print("The best hyperparameters are: ", best_hyperparameters)

    best_model = best_hyperparameters["best_model"]
    # Evaluate the best model on the test set
    best_model.eval()
    pred = best_model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    test_acc = int(correct) / int(data.test_mask.sum())
    print(f'Test accuracy with best hyperparameters: {test_acc:.4f}')

    # Save the best model
    save_dir = os.path.join(parent_directory, 'save_model/nodegcn')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'node_gcn_best_model.pt'))





