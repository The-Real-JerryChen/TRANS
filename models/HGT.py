import torch
from torch_geometric.nn import Linear
from layers.HGC import *

def get_bounds_from_slice_dict(batch_obj):
    if not hasattr(batch_obj, '_slice_dict'):
        raise RuntimeError("The batch object does not have _slice_dict attribute.")
    key = 'visit'  
    slices = batch_obj._slice_dict[key]
    return slices['x']

def get_last_visit_features_from_slices(x, slices_tensor):
    last_visit_features = []
    for idx in range(slices_tensor.size(0) - 1):
        start, end = int(slices_tensor[idx]), int(slices_tensor[idx + 1])
        last_visit_feature = x[end - 1]  
        last_visit_features.append(last_visit_feature)
    return torch.stack(last_visit_features)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, edge_index_dict, batch):        
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in batch.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, batch.edge_time_dict)
        s = get_bounds_from_slice_dict(batch)
        tmp  = get_last_visit_features_from_slices(x_dict['visit'],s)

        return self.lin(tmp)
    
