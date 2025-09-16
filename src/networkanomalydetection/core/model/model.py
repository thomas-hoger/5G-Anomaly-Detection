import torch
from torch_geometric.nn import GINEConv, global_mean_pool


class MyGNN(torch.nn.Module):

    def __init__(self, in_node_feats, in_edge_feats, hidden_dim, num_layers, out_dim):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        self.convs.append(GINEConv(
            nn=torch.nn.Sequential(torch.nn.Linear(in_node_feats + in_edge_feats, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(hidden_dim, hidden_dim))
        ))

        for _ in range(num_layers-1):
            self.convs.append(GINEConv(
                nn=torch.nn.Sequential(torch.nn.Linear(hidden_dim + in_edge_feats, hidden_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_dim, hidden_dim))
            ))

        self.lin = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)
