import os
import torch
import torch.nn as nn
import ray
from tqdm import tqdm
from torch_geometric.transforms import NormalizeFeatures
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint
from gnn_graph_classification.graph_gnn import GraphGNN
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def train_and_validate(config):
    # In GNN, one graph is considered as a data point when doing batch stochastic gradient, since all nodes in the graph
    # somehow connected. However, the dataset only contains one graph, so we don't need to worry about the batch in this
    # case.
    train_loader = ray.get(train_loader_id)
    val_loader = ray.get(val_loader_id)
    gcn_dim = config["gnn_dim"]
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    num_epochs = config["num_epochs"]

    model = GraphGNN(in_features, num_classes, gcn_dim, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    early_stop_threshold = 0.5

    hyperparameters_str = '_'.join(f"{k}={v}" for k, v in config.items())
    save_dir = os.path.join(SAVE_DIR, hyperparameters_str)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        correct_train_pred_epoch = 0
        for train_data in train_loader:
            optimizer.zero_grad()
            out = model(train_data)
            loss = criterion(out, train_data.y)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            train_prediction = out.argmax(dim=1)
            correct_train_pred_epoch += (train_prediction == train_data.y).sum()
        train_accuracy = int(correct_train_pred_epoch) / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            correct_val_pred_epoch = 0
            for val_data in val_loader:
                prediction = model(val_data).argmax(dim=1)
                correct_val_pred_epoch += (prediction == val_data.y).sum()
            val_accuracy = int(correct_val_pred_epoch) / len(val_loader.dataset)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(save_dir, f"model.pth"))

            # Early stopping condition
            if train_accuracy - val_accuracy > early_stop_threshold and epoch >= num_epochs // 2:
                print("Early stopping due to overfitting")
                break

    train.report(metrics={"best_val_accuracy": best_val_accuracy},  checkpoint=Checkpoint(path=save_dir))


if __name__ == "__main__":
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    SAVE_DIR = os.path.join(parent_directory, 'save_model/graphgnn')
    download_path = os.path.join(parent_directory, 'data')
    dataset = TUDataset(root=download_path, name='MUTAG', transform=NormalizeFeatures())
    in_features = dataset.num_node_features
    num_classes = dataset.num_classes
    print(dataset[0])

    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
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

    ray.init()
    config = {
        "lr": tune.uniform(1e-5, 1e-2),
        "weight_decay": tune.uniform(1e-6, 1e-4),
        "gnn_dim": tune.choice([[i] for i in range(50, 100)]),
        "dropout_p": tune.uniform(0.3, 0.7),
        "num_epochs": tune.choice([200, 300])
    }

    tuner = tune.Tuner(
        train_and_validate,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=ASHAScheduler(metric="best_val_accuracy", mode="max"),
        ),
    )

    # Put large objects in the Ray object store
    dataset_id = ray.put(dataset)
    train_loader_id = ray.put(train_loader)
    val_loader_id = ray.put(val_loader)

    results = tuner.fit()

    best_result = results.get_best_result("best_val_accuracy", "max", "all")
    print(f"Best trial config: {best_result}")
    print(f"Best trial final validation accuracy: {best_result.metrics['best_val_accuracy']}")
    best_config = best_result.config
    print(f"Best config hyperparameters: {best_config}")

    # Load the best model
    best_model_dir = best_result.checkpoint.path
    best_model_state_dict = torch.load(os.path.join(best_model_dir, "model.pth"))
    best_model = GraphGNN(in_features, num_classes, best_config["gnn_dim"], best_config["dropout_p"]).to(device)
    best_model.load_state_dict(best_model_state_dict)

    best_model.eval()
    test_correct = 0
    for test_data in test_loader:
        test_pred = best_model(test_data).argmax(dim=1)
        test_correct += (test_pred == test_data.y).sum()
    test_acc = int(test_correct) / len(test_loader.dataset)
    print(f'Test accuracy with best hyperparameters: {test_acc:.4f}')
