import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv


class GINEEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()

        # ----------- MLP pour la première GINEConv -----------
        mlp1 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINEConv(
            nn=mlp1,
            edge_dim=edge_dim
        )

        # ----------- MLP pour la deuxième GINEConv -----------
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.conv2 = GINEConv(
            nn=mlp2,
            edge_dim=edge_dim
        )

        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm(x)
        return x

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z, edge_index):
        row, col = edge_index
        pair = torch.cat([z[row], z[col]], dim=-1)
        return self.mlp(pair).squeeze(-1)

class BaselineAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.edge_attr)
        adj_pred = self.decoder(z, data.edge_index)
        return adj_pred
