import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn import (
    ModuleList,
    Linear,
)
from torch.nn.functional import dropout, relu

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
    DataLoader,
    NeighborSampler,
)
from torch_geometric.utils import (
    coalesce,
    to_networkx,
    train_test_split_edges,
    add_self_loops,
    degree,
)
from torch_geometric.nn import (
    GAT,
    GATConv,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
    MessagePassing,
)

import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.loader import NeighborLoader, DataLoader
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 32
num_features = 5
hid_channels = 32
out_channels = 32
# Graph attention network
class GAT(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GAT, self).__init__()
        torch.manual_seed(42)
        # GAT layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features,embedding_size, heads=8, dropout=0.2))
        self.convs.append(GATConv(embedding_size *8, hid_channels, heads=2, dropout=0.2))
        self.convs.append(GATConv(hid_channels *2, out_channels, heads=1, dropout=0.2))
        # Output layer
        self.out = Linear(out_channels*2, 1)

    def forward(self, x, edge_index, batch_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.tanh(x)
        # Apply a final (linear) classifier.
        x = torch.cat([gmp(x, batch_index),
                            gap(x, batch_index)], dim=1)
        out = self.out(x)
        return out, x
