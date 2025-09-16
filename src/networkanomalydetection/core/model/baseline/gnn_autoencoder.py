import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GNNAutoEncoder(nn.Module):

    def __init__(self, feature_size: int, hidden_dims: list = [512, 256, 128]):
        super().__init__()

        self.feature_size = feature_size
        self.hidden_dims  = hidden_dims

    def encode(self,x,edge_index):

        dims = [self.feature_size] + self.hidden_dims
        for i in range(len(dims) - 1):

            x = SAGEConv(dims[i], dims[i+1], aggr='mean')(x,edge_index)
            x = nn.ReLU()(x)
            x = nn.Dropout(0.2)(x)

        return x

    def decode(self, x):

        dims = [self.feature_size] + self.hidden_dims
        dims.reverse()

        for i in range(len(dims) - 1):

            x = nn.Linear(dims[i], dims[i+1])(x)
            x = nn.ReLU()(x)
            x = nn.Dropout(0.1)(x)

        return x


    def forward(self, data):

        node_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        z = self.encode(node_attr)
        return self.decode(z)

# https://github.com/deepfindr/gvae/blob/master/gvae.py