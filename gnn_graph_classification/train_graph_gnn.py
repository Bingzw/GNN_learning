import os
import torch
import torch.nn as nn
from tqdm import tqdm
from gnn_graph_classification.graph_gnn import GraphGNN
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures


def train_and_validate(train_loader, val_loader, model, optimizer, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(num_epochs)):
        for train_data in train_loader:
            optimizer.zero_grad()
            out = model(train_data)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct_pred = 0
            for val_data in val_loader:
                prediction = model(val_data).argmax(dim=1)
                correct_pred = (prediction == val_data.y).sum()
            val_accuracy = int(correct_pred) / len(val_loader.dataset)

    return model, val_accuracy


if __name__ == "__main__":
    seed = 42
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    download_path = os.path.join(parent_directory, 'data')
    dataset = TUDataset(root=download_path, name='MUTAG', transform=NormalizeFeatures())
    print(dataset[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.6)
    val_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - val_size
    # split train, validation and test dataset
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    # load data
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # each batch would load data like this:
    # DataBatch(edge_index=[2, 2636], x=[1188, 7], edge_attr=[2636, 4], y=[64], batch=[1188], ptr=[65])
    # batch is a vector that maps each node to its respective graph, and ptr is a vector that each value i points to the
    # cumulative sum of the number of nodes in the first i graphs in the batch

    in_features = dataset.num_node_features
    num_classes = dataset.num_classes
    # hyperparameters
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
                    model = GraphGNN(in_features, num_classes, dim, dp).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                    model, val_acc = train_and_validate(train_loader, val_loader, model, optimizer, num_epochs)
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
    test_correct = 0
    for test_data in test_loader:
        test_pred = best_model(test_data).argmax(dim=1)
        test_correct = (test_pred == test_data.y).sum()
    test_acc = int(test_correct) / len(test_loader.dataset)
    print(f'Test accuracy with best hyperparameters: {test_acc:.4f}')

    # Save the best model
    save_dir = os.path.join(parent_directory, 'save_model/graphgnn')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(save_dir, 'graph_gnn_best_model.pt'))
