import torch
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.conv import GATConv 
from torch.nn.functional import softmax, sigmoid, relu
from torch.nn import Parameter, ModuleList, BatchNorm1d
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import MessagePassing, global_add_pool
class GAT(torch.nn.Module):
    def __init__(self, in_channels, num_gnn_layers, num_linear_layers, linear_out_channels):
        super(GAT, self).__init__()
        self.in_channels = in_channels

        self.num_gnn_layers = num_gnn_layers
        self.attention_layers = ModuleList()
        for layer_id in range(num_gnn_layers):
            self.attention_layers.append(GATConv(in_channels, in_channels, True))
        self.pooling_layer = global_add_pool

        self.num_linear_layers = num_linear_layers
        self.linear_layers = ModuleList()
        for layer_id in range(num_linear_layers):
            if layer_id == 0:
                self.linear_layers.append(Linear(self.in_channels, linear_out_channels[0]))
                continue

            self.linear_layers.append(Linear(linear_out_channels[layer_id - 1], linear_out_channels[layer_id]))

        self.out = Linear(linear_out_channels[-1], 1)

    def forward(self, batched_data):
        x, edge_index_1, edge_index_2, edge_weight = batched_data["x"], batched_data["edge_index_1"], batched_data["edge_index_2"], batched_data["edge_weight"]

        for index,layer in enumerate(self.attention_layers):
          if index == 0:
                x_1 = layer(x, edge_index_1)
                x_2 = layer(x, edge_index_2, edge_weight)
                continue
          x_1 = layer(x_1, edge_index_1)
          x_2 = layer(x_2, edge_index_2, edge_weight)

        node_representation = x_2 - x_1
        graph_representation = self.pooling_layer(node_representation, batched_data.batch)
        
        for layer in self.linear_layers:
            graph_representation = layer(graph_representation)
            graph_representation = relu(graph_representation)

        binding_affinity = self.out(graph_representation)

        return torch.nan_to_num(binding_affinity)
