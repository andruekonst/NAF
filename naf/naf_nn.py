import numpy as np
import torch
from torch.nn import Module, Sequential, Linear, Tanh


# NEG_INF = float('-inf')
NEG_INF = -1.e20


def make_encoder(n_features: int, hidden_size: int, n_layers: int):
    layers = [Linear(n_features, hidden_size, dtype=torch.double)]
    for i in range((n_layers - 1) * 2):
        if i % 2 == 0:
            layers.append(Tanh())
        else:
            layers.append(Linear(hidden_size, hidden_size, dtype=torch.double))
    return Sequential(*layers)


class NAFLeafNetwork(Module):
    def __init__(self, n_features: int, hidden_size: int = 16, n_layers: int = 1):
        super().__init__()
        self.first_encoder = make_encoder(n_features, hidden_size, n_layers)
        self.neighbors_count_ = None
        self.neighbors_hot_id_ = None

    def _get_neighbors_count(self, neighbors_hot):
        # cached calculation of neighbors count
        if self.neighbors_hot_id_ != id(neighbors_hot):
            self.neighbors_count_ = torch.sum(neighbors_hot, dim=1)
            self.neighbors_hot_id_ = id(neighbors_hot)
        return self.neighbors_count_


    def forward(self, X, background_X, background_y, neighbors_hot):
        """
        Args:
            X: Input points (keys) of shape (n_samples, n_features).
            background_X: Data points inside the leaves of shape (n_background, n_features).
            background_y: Target value of data points inside the leaves.
            neighbors_hot: Corresponding leaf data points hot representation of shape (n_samples, n_background, n_trees).
        """
        assert len(background_y.shape) == 2
        # neighbors_count = self._get_neighbors_count(neighbors_hot)  # shape: (n_samples, n_trees)
        X_enc = self.first_encoder(X)
        background_X_enc = self.first_encoder(background_X)
        dots = X_enc @ background_X_enc.T  # shape: (n_samples, n_background)
        n_trees = neighbors_hot.shape[2]
        dots_trees = dots.unsqueeze(-1).repeat(1, 1, n_trees)  # shape: (n_samples, n_background, n_trees)
        dots_trees[~neighbors_hot] = NEG_INF
        # the first attention: attend to each data point inside a leaf
        first_alphas = torch.softmax(dots_trees, dim=1)  # shape: (n_samples, n_background, n_trees)
        # one_neighbor_mask = (neighbors_count == 1)[:, np.newaxis, :]
        first_leaf_xs = torch.einsum('sbt,bf->stf', first_alphas, background_X)  # shape: (n_samples, n_trees, n_features)
        first_leaf_y = torch.einsum('sbt,by->sty', first_alphas, background_y)  # shape: (n_samples, n_trees, n_out)
        return first_leaf_xs, first_leaf_y, first_alphas


class NAFTreeNetwork(Module):
    def __init__(self, n_features: int, hidden_size: int = 16, n_layers: int = 1):
        super().__init__()
        self.second_encoder = make_encoder(n_features, hidden_size, n_layers)

    def forward(self, X, first_leaf_xs, first_leaf_y, first_alphas, need_attention_weights: bool = False):
        """
        Args:
            X: Input points (keys) of shape (n_samples, n_features).
            first_leaf_xs: Leaf xs.
            first_leaf_y: Leaf y.
            first_alphas: First attention weights.
            need_attention_weights: Use attention weights or not.
        """
        # samples, trees, features
        # the second attention: over trees
        second_X_enc = self.second_encoder(X)
        second_leaf_xs_enc = self.second_encoder(first_leaf_xs.view(-1, first_leaf_xs.shape[2]))\
                                 .view(first_leaf_xs.shape[0], first_leaf_xs.shape[1], -1)
        # second_leaf_xs_enc = self.second_encoder(first_leaf_xs)
        second_dots = torch.einsum('nf,ntf->nt', second_X_enc, second_leaf_xs_enc)  # shape: (n_samples, n_trees)
        second_betas = torch.softmax(second_dots, dim=1)  # shape: (n_samples, n_trees)
        second_y = torch.einsum('nty,nt->ny', first_leaf_y, second_betas)
        if need_attention_weights:
            second_xs = torch.einsum('ntf,nt->nf', first_leaf_xs, second_betas)
            return second_y, second_xs, first_alphas, second_betas
        return second_y


class NAFNetwork(Module):
    def __init__(self, n_features: int, hidden_size: int = 16, n_layers: int = 1):
        super().__init__()
        self.leaf_network = NAFLeafNetwork(n_features, hidden_size, n_layers)
        self.tree_network = NAFTreeNetwork(n_features, hidden_size, n_layers)

    def forward(self, X, background_X, background_y, neighbors_hot, need_attention_weights: bool = False):
        """
        Args:
            X: Input points (keys) of shape (n_samples, n_features).
            background_X: Data points inside the leaves of shape (n_background, n_features).
            background_y: Target value of data points inside the leaves.
            neighbors_hot: Corresponding leaf data points hot representation of shape (n_samples, n_background, n_trees).
        """
        first_leaf_xs, first_leaf_y, first_alphas = self.leaf_network(X, background_X, background_y, neighbors_hot)
        return self.tree_network(X, first_leaf_xs, first_leaf_y, first_alphas, need_attention_weights)

