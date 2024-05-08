import torch
from torch_geometric.data import HeteroData
from layers.TSEncoder import *
from utils import *
from tqdm import *

class PatientGraph(torch.utils.data.Dataset):
    def __init__(self, tokenizer, subset, dim, device, trans_dim = 0, di = False):
        self.c_tokenzier = tokenizer['cond_hist']
        self.d_tokenzier = tokenizer['drugs']
        self.p_tokenzier = tokenizer['procedures']
        self.dataset = subset
        self.di_edge = di
        self.dim = dim
        self.se = False
        if trans_dim!=0:
            self.se = True         
            self.global_pe_transform = AddGlobalLaplacianPE(k=trans_dim, device = device)
            self.local_se_transform = AddMetaPathRandomWalkSE(trans_dim, device = device)

        self.all_data = self._process_()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self): 
        return len(self.all_data)

    def _process_(self):
        all_data = []
        print('Constructing graph...')
        for dp in tqdm(range(len(self.dataset))):
            data = HeteroData()
            num_visit = len(self.dataset[dp]['procedures'])
            data['visit'].x = torch.zeros(num_visit, self.dim) #self.visits[dp].unsqueeze(0).expand(num_visit, -1) #torch.zeros(num_visit, self.dim)
            dpc = self.c_tokenzier.batch_encode_2d(self.dataset[dp]['cond_hist'], padding=False)
            dpp = self.p_tokenzier.batch_encode_2d(self.dataset[dp]['procedures'], padding=False)
            dpd = self.d_tokenzier.batch_encode_2d(self.dataset[dp]['drugs'], padding=False)

            data['visit'].time = convert_to_relative_time(self.dataset[dp]['adm_time'])

            data['co'].x = torch.zeros(self.c_tokenzier.get_vocabulary_size(),self.dim)
            data['pr'].x = torch.zeros(self.p_tokenzier.get_vocabulary_size(),self.dim)
            data['dh'].x = torch.zeros(self.d_tokenzier.get_vocabulary_size(),self.dim)

            civ =  torch.tensor([[item for sublist in dpc for item in sublist],
                                                    [index for index, sublist in enumerate(dpc) for _ in sublist]], dtype=torch.int64)
            data['co', 'in', 'visit'].edge_index = civ
            piv =torch.tensor( [[item for sublist in dpp for item in sublist],
                                                    [index for index, sublist in enumerate(dpp) for _ in sublist]], dtype=torch.int64)
            data['pr', 'in', 'visit'].edge_index = piv
            div = torch.tensor([[item for sublist in dpd for item in sublist],
                                                    [index for index, sublist in enumerate(dpd) for _ in sublist]], dtype=torch.int64)
            data['dh', 'in', 'visit'].edge_index = div
            viv = torch.tensor([[i for i in range(num_visit-1)], [i+1 for i in range(num_visit-1)]], dtype=torch.int64)
            data['visit','connect','visit'].edge_index = viv

            fciv = torch.flip(civ, [0])
            fpiv = torch.flip(piv, [0])
            fdiv = torch.flip(div, [0])

            data['co', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpc) for _ in sublist], dtype=torch.float32)
            data['pr', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpp) for _ in sublist], dtype=torch.float32)
            data['dh', 'in', 'visit'].edge_time = torch.tensor([index for index, sublist in enumerate(dpd) for _ in sublist], dtype=torch.float32)

            data['visit', 'has', 'co'].edge_index = fciv
            data['visit', 'has', 'pr'].edge_index = fpiv
            data['visit', 'has', 'dh'].edge_index = fdiv

            if self.di_edge:
                data['visit','connect','visit'].edge_index = viv

            else:
                data['visit','connect','visit'].edge_index  = torch.cat([viv, torch.flip(viv, [0])], dim=1)

            if self.se:
                data = self.global_pe_transform.apply_laplacian_pe(data)
                f_metapaths = [[('co', 'in', 'visit'), ('visit', 'has', 'pr'),("pr",'in', "visit"), ('visit', 'has', 'co')],
                [("pr",'in', "visit"), ('visit', 'has', 'co'),('co', 'in', 'visit'), ('visit', 'has', 'pr')  ],
                [("dh",'in', "visit"), ('visit', 'has', 'dh')]
                ]
                data = self.local_se_transform.forward(data = data, metapaths= f_metapaths)
                for node_type in data.node_types:
                    if node_type not in ['co', 'pr', 'dh']:
                        del data[node_type].laplacian_pe
                        del data[node_type].random_walk_se

            all_data.append(data)
        return all_data
