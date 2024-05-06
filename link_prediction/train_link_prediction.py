import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm import tqdm
from heterdata_creation import create_heter_movie_rating_data
from torch_geometric.loader import LinkNeighborLoader
from heter_gnn import RatingHeterGNNModel
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    SAVE_DIR = os.path.join(parent_directory, 'save_model/linkpred')
    movies_path = os.path.join(parent_directory, 'data/ml-latest-small/movies.csv')
    ratings_path = os.path.join(parent_directory, 'data/ml-latest-small/ratings.csv')
    data = create_heter_movie_rating_data(movies_path, ratings_path)
    print(data)

    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing (to update node embeddings)
    # and 30% of edges for supervision (to calculate the loss and update the model parameters).
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly, so we don't want to
    # add them to the graph right away.
    num_edges = data["user", "rates", "movie"].num_edges
    transform = T.RandomLinkSplit(
        num_val=int(num_edges * 0.1),
        num_test=int(num_edges * 0.1),
        disjoint_train_ratio=0.3,  # 30% of edges for supervision, 70% for message passing
        neg_sampling_ratio=2,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "movie"),
        rev_edge_types=("movie", "rev_rates", "user"),
    )

    train_data, val_data, test_data = transform(data)

    # define minibatch loaders to generate subgraphs for training.
    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.

    # Define seed edges:
    edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    edge_label = train_data["user", "rates", "movie"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),  # The edge indices for which neighbors are
        # sampled to create mini-batches
        edge_label=edge_label,  # The labels of edge indices for which neighbors are sampled
        batch_size=128,
        shuffle=True,
    )

    hidden_dim = 64
    num_users = data["user"].num_nodes
    num_movies = data["movie"].num_nodes
    metadata = data.metadata()

    model = RatingHeterGNNModel(hidden_dim, num_users, num_movies, metadata).to(device)
    print(model)
    # train a hetergeneous link level GNN model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()

            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            y = sampled_data["user", "rates", "movie"].edge_label.float().to(device)
            loss = F.binary_cross_entropy_with_logits(pred, y)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # Define the validation seed edges:
    edge_label_index = val_data["user", "rates", "movie"].edge_label_index
    edge_label = val_data["user", "rates", "movie"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )

    preds = []
    ground_truths = []
    for sampled_data in tqdm(val_loader):
        with torch.no_grad():
            # Run the forward pass of the model
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            # Append the predictions to `preds`
            preds.append(pred)
            # Get the ground-truth labels
            y = sampled_data["user", "rates", "movie"].edge_label.float().to(device)
            # Append the ground-truth labels to `ground_truths`
            ground_truths.append(y)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")