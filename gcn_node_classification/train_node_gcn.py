import os
import torch
import torch.nn as nn
import ray
from tqdm import tqdm
from gcn_node_classification.node_gcn import NodeGCN
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint


def train_and_validate(config):
    # In GNN, one graph is considered as a data point when doing batch stochastic gradient, since all nodes in the graph
    # somehow connected. However, the dataset only contains one graph, so we don't need to worry about the batch in this
    # case.
    data = dataset[0].to(device)
    gcn_dim = config["gcn_dim"]
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    num_epochs = config["num_epochs"]

    model = NodeGCN(in_features, num_classes, gcn_dim, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    early_stop_threshold = 0.3

    hyperparameters_str = '_'.join(f"{k}={v}" for k, v in config.items())
    save_dir = os.path.join(SAVE_DIR, hyperparameters_str)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        train_prediction = out[data.train_mask].argmax(dim=1)
        correct_train_pred = (train_prediction == data.y[data.train_mask]).sum()
        train_accuracy = int(correct_train_pred) / int(data.train_mask.sum())

        model.eval()
        with torch.no_grad():
            prediction = model(data).argmax(dim=1)
            correct_pred = (prediction[data.val_mask] == data.y[data.val_mask]).sum()
            val_accuracy = int(correct_pred) / int(data.val_mask.sum())

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(save_dir, f"model.pth"))

            # Early stopping condition
            if train_accuracy - val_accuracy > early_stop_threshold and epoch >= num_epochs // 4:
                print("Early stopping due to overfitting")
                break

    train.report(metrics={"best_val_accuracy": best_val_accuracy},  checkpoint=Checkpoint(path=save_dir))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    SAVE_DIR = os.path.join(parent_directory, 'save_model/nodegcn')
    download_path = os.path.join(parent_directory, 'data')
    dataset = Planetoid(root=download_path, name='Cora', transform=NormalizeFeatures())
    in_features = dataset.num_node_features
    num_classes = dataset.num_classes
    print(dataset[0])

    ray.init()
    config = {
        "lr": tune.uniform(1e-5, 1e-2),
        "weight_decay": tune.uniform(1e-6, 1e-4),
        "gcn_dim": tune.choice([[i] for i in range(4, 21)]),
        "dropout_p": tune.uniform(0.3, 0.7),
        "num_epochs": tune.choice([100, 200, 300])
    }

    tuner = tune.Tuner(
        train_and_validate,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=ASHAScheduler(metric="best_val_accuracy", mode="max"),
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result("best_val_accuracy", "max", "all")
    print(f"Best trial config: {best_result}")
    print(f"Best trial final validation accuracy: {best_result.metrics['best_val_accuracy']}")
    best_config = best_result.config
    print(f"Best config hyperparameters: {best_config}")

    # Load the best model
    best_model_dir = best_result.checkpoint.path
    best_model_state_dict = torch.load(os.path.join(best_model_dir, "model.pth"))
    best_model = NodeGCN(in_features, num_classes, best_config["gcn_dim"], best_config["dropout_p"]).to(device)
    best_model.load_state_dict(best_model_state_dict)

    best_model.eval()
    test_data = dataset[0].to(device)
    prediction = best_model(test_data).argmax(dim=1)
    correct_pred = (prediction[test_data.test_mask] == test_data.y[test_data.test_mask]).sum()
    test_accuracy = int(correct_pred) / int(test_data.test_mask.sum())
    print(f'Test accuracy with best hyperparameters: {test_accuracy:.4f}')





