import torch
from torch import nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gat1 = GATConv(in_channels, 16, heads=4, dropout=0.3)
        self.out = nn.Linear(16 * 4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = torch.nn.functional.elu(x)
        return self.out(x).squeeze()

