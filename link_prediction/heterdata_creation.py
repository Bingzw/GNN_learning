import torch
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData


def create_heter_movie_rating_data(movies_path, ratings_path):
    """
    Create a heterogeneous graph from the MovieLens dataset.
    :param movies_path: the path to the movies.csv file
    :param ratings_path: the path to the ratings.csv file
    :return: heterogeneous graph data object
    """
    movies_df = pd.read_csv(movies_path, index_col='movieId')
    genres = movies_df['genres'].str.get_dummies('|')
    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)
    ratings_df = pd.read_csv(ratings_path)
    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = pd.DataFrame(data={
        'movieId': movies_df.index,
        'mappedID': pd.RangeIndex(len(movies_df)),
    })
    # Perform merge to obtain the edges from users and movies:
    ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                               left_on='userId', right_on='userId', how='left')
    ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
    ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                                left_on='movieId', right_on='movieId', how='left')
    ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)
    # Cconstruct our `edge_index` in COO format
    edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)

    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(movies_df))

    # Add the node features and edge indices:
    data["movie"].x = movie_feat
    data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the `T.ToUndirected()` transform for this from PyG:

    # Add reverse edges from "movie" to "user"
    data = T.ToUndirected()(data)

    return data