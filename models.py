import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseHyperGraphAttentionLayer, RankingLayer

class HyperGAT(nn.Module):
    def __init__(self, entity_embs, relation_embs, max_arity, data_arity, nfeat, nhid, nemb, alpha, dropout, nheads, device):
        super().__init__()
        self.emb_dim = nemb
        self.dropout = dropout
        self.device = device
        self.entity_embs = nn.Parameter(entity_embs)
        self.relation_embs = nn.Parameter(relation_embs)
       
        self.attentions = [SparseHyperGraphAttentionLayer(nfeat, nhid, nemb, max_arity, alpha, dropout, device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)
        self.out_attention = SparseHyperGraphAttentionLayer(nhid * nheads, nemb, nemb, max_arity, alpha, dropout, device, concat=False)
        self.scoring = RankingLayer(data_arity, dropout)

    def forward(self, batch_inputs, tuples, H, node_indices, edge_indices):
        out_entity_embs = torch.cat([attention(self.entity_embs, self.relation_embs, tuples, H, node_indices, edge_indices) for attention in self.attentions], dim=1)
        out_entity_embs = F.dropout(out_entity_embs, self.dropout, training=self.training)
        out_entity_embs = F.elu(self.out_attention(out_entity_embs, self.relation_embs, tuples, H, node_indices, edge_indices))

        batch_outputs = torch.ones(batch_inputs.shape).unsqueeze(2).repeat_interleave(self.emb_dim, 2).to(self.device)
        batch_outputs[:, 0, :] = self.relation_embs[batch_inputs[:, 0]-1, :]
        batch_outputs[:, 1:, :] = torch.where(batch_inputs.unsqueeze(2)[:, 1:, :] > 0, out_entity_embs[batch_inputs[:, 1:]-1], batch_outputs[:, 1:, :])
        batch_scores = self.scoring(batch_outputs)
        return batch_scores

