import torch
import torch.nn as  nn
import math
from typing import Optional,  Tuple, List
from Seqmodels import *
from pyhealth.medcode import InnerMap
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence

class Transformer(nn.Module):
    def __init__(
        self,
        Tokenizers,
        output_size,
        device,
        embedding_dim = 128,
        dropout = 0.5
    ):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                feature_size=embedding_dim, dropout = dropout,num_layers=2
            )
        
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata) :
        patient_emb = []
        for feature_key in self.feature_keys:

            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
                # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits
    
class RETAIN(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
        embedding_dim: int = 128, dropout = 0.5
        ):
        super(RETAIN, self).__init__()
        self.embedding_dim = embedding_dim
        Tokenizers =  {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout = dropout)

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.device = device


    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):

        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.sum(x, dim=2) != 0
            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits
    
class StageNet(nn.Module):
    def __init__(self, Tokenizers, output_size, device, embedding_dim: int = 128,
        chunk_size: int = 128,
        levels: int = 3,
    ):
        super(StageNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels
        Tokenizers =  {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feature_keys = Tokenizers.keys()

        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

        self.stagenet = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
            )
        self.fc = nn.Linear(
            len(self.feature_keys) * self.chunk_size * self.levels, output_size
        )
        self.device = device

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

    def forward(self, batchdata):
        patient_emb = []
        distance = []
        mask_dict = {}
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
 
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x !=0, dim=2)
            mask_dict[feature_key] = mask
            time = None
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits

class KAME(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
        embedding_dim: int = 128, dataset = 'mimic3'
        ):
        super(KAME, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.parent_dictionary = {'cond_hist':InnerMap.load("ICD9CM"), 'procedures':InnerMap.load("ICD9PROC")}
        self.compatability = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False),
        )
        
        self.knowledge_map = nn.ModuleDict()
        for feature_key in ['cond_hist', 'procedures']:
            self.knowledge_map[feature_key] = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        
        self.rnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            if feature_key.endswith('_parent'):
                continue
            self.rnn[feature_key] = nn.GRU(input_size=self.embedding_dim, 
                                           hidden_size=self.embedding_dim, 
                                           batch_first=True, bidirectional=False)
        
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def embed_code_with_parent(self, x, feature_key):
        # x: (patient, visit, event)
        max_visit = x.shape[1]
        out = []
        out_mask = []
        for patient in x:
            mask = []
            patient_embed = []
            for visit in patient:
                if visit.sum() == 0:
                    num_pad = max_visit - len(patient_embed)
                    mask.extend([0] * num_pad)
                    visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                    patient_embed.extend([visit_embed] * num_pad)
                    break
                visit = visit[visit != 0]
                mask.append(1)
                events = self.feat_tokenizers[feature_key].convert_indices_to_tokens(visit.tolist())
                basic_embeds = self.embeddings[feature_key](visit)
                visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                for embed, event in zip(basic_embeds, events):
                    try:
                        parents = self.parent_dictionary[feature_key].get_ancestors(event)
                    except:
                        visit_embed += embed
                        continue
                    parents = self.feat_tokenizers[feature_key + '_parent'].convert_tokens_to_indices(parents)
                    parents = torch.tensor(parents, dtype=torch.long, device=self.device)
                    parents_embed = self.embeddings[feature_key + '_parent'](parents)
                    parents_embed = torch.cat([parents_embed, embed.reshape(1,-1)], dim=0)
                    embed_ = torch.stack([embed] * len(parents_embed))
                    compat_score = self.compatability(torch.cat([embed_, parents_embed], dim=1))
                    compat_score = torch.softmax(compat_score, dim=0)
                    embed = torch.sum(compat_score * parents_embed, dim=0)
                    visit_embed += embed
                patient_embed.append(visit_embed)
            patient_embed = torch.stack(patient_embed)
            out.append(patient_embed)
            out_mask.append(mask)
        out = torch.stack(out)
        out_mask = torch.tensor(out_mask, dtype=torch.int, device=self.device)
        return out, out_mask

    def embed_code(self, x, feature_key):
        # x: (patient, visit, event)
        max_visit = x.shape[1]
        out = []
        out_mask = []
        for patient in x:
            mask = []
            patient_embed = []
            for visit in patient:
                if visit.sum() == 0:
                    num_pad = max_visit - len(patient_embed)
                    mask.extend([0] * num_pad)
                    visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                    patient_embed.extend([visit_embed] * num_pad)
                    break
                visit = visit[visit != 0]
                mask.append(1)
                embeds = self.embeddings[feature_key](visit)
                visit_embed = torch.sum(embeds, dim=0)
                patient_embed.append(visit_embed)
            patient_embed = torch.stack(patient_embed)
            out.append(patient_embed)
            out_mask.append(mask)
        out = torch.stack(out)
        out_mask = torch.tensor(out_mask, dtype=torch.int, device=self.device)
        return out, out_mask

    def get_parent_embeddings(self, x, feature_key):
        out = []
        for patient in x:
            if patient == []:
                out.append(torch.zeros(self.embedding_dim, device=self.device))
                continue
            parent = set()
            for code in patient:
                try:
                    parent.update(self.parent_dictionary[feature_key].get_ancestors(code))
                except:
                    continue
            parent = list(parent)
            parent = self.feat_tokenizers[feature_key + '_parent'].convert_tokens_to_indices(parent)
            parent = torch.tensor(parent, dtype=torch.long, device=self.device)
            parent = self.embeddings[feature_key + '_parent'](parent)
            out.append(parent)
        return out
    
    def forward(self, batchdata):
        patient_emb = []
        patient_parent = {}
        for feature_key in self.feature_keys:
            if feature_key.endswith('_parent'):
                continue
            if feature_key != 'drugs':
                if feature_key == 'cond_hist':
                    x = list(map(lambda y: y[-2] if len(y) > 1 else y[-1], batchdata[feature_key]))
                else:
                    x = list(map(lambda y: y[-1], batchdata[feature_key]))
                patient_parent[feature_key] = self.get_parent_embeddings(x, feature_key)

            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, embedding_dim)
            if feature_key != 'drugs':
                x, mask = self.embed_code_with_parent(x, feature_key)
            else:
                x, mask = self.embed_code(x, feature_key)
            
            visit_len = mask.sum(dim=1)
            visit_len[visit_len == 0] = 1
            visit_len = visit_len.cpu()
            x = pack_padded_sequence(x, visit_len, batch_first=True, enforce_sorted=False)
            x, _ = self.rnn[feature_key](x)
            x = unpack_sequence(x)
            x = list(map(lambda x: x[-1], x))
            x = torch.stack(x)
            mask = (mask.sum(dim=1).reshape(-1,1) != 0)
            x = x * mask
            patient_emb.append(x)
        
        tmp_patient_emb = torch.sum(torch.stack(patient_emb), dim=0)
        for key in patient_parent.keys():
            knowledge_embed = patient_parent[key]
            mask = list(map(lambda x: 0 if (x == 0).all() else 1, knowledge_embed))
            knowledge_embed = [self.knowledge_map[key](x) for x in knowledge_embed]
            patient_knowledge_embed = []
            for patient, basic_embed, mask_ in zip(knowledge_embed, tmp_patient_emb, mask):
                if mask_ == 0:
                    patient_knowledge_embed.append(torch.zeros(self.embedding_dim, device=self.device))
                    continue
                weight = torch.matmul(patient, basic_embed)
                weight = torch.softmax(weight, dim=0).reshape(-1,1)
                patient = torch.sum(weight * patient, dim=0)
                patient_knowledge_embed.append(patient)
            patient_knowledge_embed = torch.stack(patient_knowledge_embed)
            patient_emb.append(patient_knowledge_embed)
        
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )