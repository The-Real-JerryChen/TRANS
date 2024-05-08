import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
import torch_sparse
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from torch_geometric.transforms import BaseTransform
from typing import Optional, List



class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        
        time_dim = expand_dim // 2  
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        seq_len = ts.size(0)
        ts = ts.view(seq_len, 1)
        map_ts = ts * self.basis_freq.view(1, -1)  
        map_ts += self.phase.view(1, -1)

        # Creating the time encoding with both cosine and sine parts
        harmonic = torch.cat([torch.cos(map_ts), torch.sin(map_ts)], dim=-1) 

        return harmonic
    
class Time2Vec(nn.Module):
    def __init__(self, output_size):
        super(Time2Vec, self).__init__()
        self.output_size = output_size//2
        self.linear = nn.Linear(1, self.output_size)  
        self.periodic = nn.Linear(1, self.output_size)  

    def forward(self, x):
        x = x.view(-1, 1)  
        linear_out = self.linear(x)
        periodic_out = torch.sin(self.periodic(x))
        return torch.cat([linear_out, periodic_out], dim=-1).view(-1, self.output_size * 2)
    

class AddMetaPathRandomWalkSE(BaseTransform):
    def __init__(self, walk_length: int, device,  attr_name: Optional[str] = 'random_walk_se'):
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.device = device

    def forward(self, data: HeteroData, metapaths: List) -> HeteroData:
        for metapath in metapaths:
            adj_matrix = self.build_metapath_adj_matrix(data, metapath)
            se = self.compute_se(adj_matrix, self.walk_length)
            start_node_type = metapath[0][0]
            if self.attr_name in data[start_node_type]:
                data[start_node_type][self.attr_name] = torch.cat((data[start_node_type][self.attr_name], se), dim=1)
            else:
                data[start_node_type][self.attr_name] = se

        return data

    def build_metapath_adj_matrix(self, data, metapath) -> torch.Tensor:
        edge_type = metapath[0]
        adj_matrix = self.get_adj_matrix(data, edge_type)

        for edge_type in metapath[1:]:
            curr_adj_matrix = self.get_adj_matrix(data, edge_type)
            adj_matrix = adj_matrix @ curr_adj_matrix

        return adj_matrix

    def get_adj_matrix(self, data: HeteroData, edge_type):
        edge_index = data[edge_type].edge_index
        adj_matrix = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1],
                                            sparse_sizes=(data[edge_type[0]].num_nodes,
                                                            data[edge_type[2]].num_nodes))
        return adj_matrix


    def compute_se(self, adj_matrix: torch_sparse.SparseTensor, walk_length: int):
        se_list = [self.get_diagonal(adj_matrix)]

        walk_matrix = adj_matrix
        for _ in range(walk_length - 1):
            walk_matrix = torch_sparse.matmul(walk_matrix, adj_matrix)
            se_list.append(self.get_diagonal(walk_matrix))

        se = torch.stack(se_list, dim=1)#.to(self.device)
        return se

    def get_diagonal(self, sparse_matrix: torch_sparse.SparseTensor):
        row, col, value = sparse_matrix.coo()
        if value is None:
            # 创建一个大小等于矩阵维度的零张量
            size = sparse_matrix.size(0)
            return torch.zeros(size, dtype=value.dtype if value is not None else torch.float32)
        mask = row == col
        return value[mask]

class AddGlobalLaplacianPE:
    def __init__(self, k: int, device):
        self.k = k  # Number of eigenvectors to use
        self.device = device

    
    def compute_laplacian_eigenvectors(self, edge_index, num_nodes, epsilon=1e-3):
        row, col = edge_index.cpu().numpy()
        data = np.ones(row.shape[0])
        L = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        L = L + L.T - coo_matrix((2 * data, (row, row)), shape=(num_nodes, num_nodes))  # L = D - A

        regularization_matrix = diags([epsilon], [0], shape=L.shape)
        regularized_L = L + regularization_matrix

        eig_vals, eig_vecs = eigsh(regularized_L, k=(self.k + 1), which='SA', ncv=2*(self.k + 1), tol=1e-3)
        eig_vecs = eig_vecs[:, eig_vals.argsort()]  
        eig_vecs = eig_vecs[:, 1:self.k + 1]
        return torch.from_numpy(eig_vecs).float()

    def apply_laplacian_pe(self, data: HeteroData):
        global_edge_index = torch.cat([data[edge_type].edge_index for edge_type in data.edge_types], dim=1).to(self.device)
        num_nodes = sum(data[num_nodes].num_nodes for num_nodes in data.node_types)

        pe = self.compute_laplacian_eigenvectors(global_edge_index, num_nodes)#.to(self.device)

        start_idx = 0
        for node_type in data.node_types:
            end_idx = start_idx + data[node_type].num_nodes
            data[node_type]['laplacian_pe'] = pe[start_idx:end_idx, :]#.to(self.device)
            start_idx = end_idx
        return data