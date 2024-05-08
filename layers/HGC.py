import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import EdgeType, Metadata, NodeType, SparseTensor
from torch_geometric.utils import softmax
from TSEncoder import TimeEncode

def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group: str = "sum",
        time_dim: int = 8,
        temp: int = 2,  
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group

        self.k_lin = ModuleDict()
        self.q_lin = ModuleDict()
        self.v_lin = ModuleDict()
        self.a_lin = ModuleDict()

        self.time_encoders = ModuleDict()
        self.time_linear = ModuleDict()

        self.tmp = temp
        self.skip = ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = ParameterDict()
        self.m_rel = ParameterDict()
        self.p_rel = ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))
            self.time_encoders[edge_type] = TimeEncode(time_dim)
            self.time_linear[edge_type] = torch.nn.Linear(time_dim, 1)
        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)

        reset(self.time_linear)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]],  # Support both.
        edge_time_dict
    ) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            # edge_index, edge_time = edge_data[0], edge_data[1]
            # edge_time = edge_time_dict[edge_type]

            src_type, _, dst_type = edge_type
            if src_type != 'visit':
                edge_time = edge_time_dict[edge_type]
            else:
                edge_time = None

            edge_type = '__'.join(edge_type)
            if edge_time != None:
                time_embedding = self.time_encoders[edge_type](edge_time)
                time_factor = torch.sigmoid(self.time_linear[edge_type](time_embedding))
                time_factor = (1 + time_factor.squeeze(-1) / self.tmp)

            a_rel = self.a_rel[edge_type]
            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)

            m_rel = self.m_rel[edge_type]
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)



            if src_type == 'visit':  # If the source node type is "visit", skip the time factor
                out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                    rel=self.p_rel[edge_type], time_factor = None, size=None)          
            else:
                out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v, 
                                    rel=self.p_rel[edge_type], time_factor=time_factor, size=None)

            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, 
                time_factor,
                ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel

        if time_factor is not None:
            alpha = alpha * time_factor.unsqueeze(-1)
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')