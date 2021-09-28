import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseHyperGraphAttentionLayer

class HyperGAT(nn.Module):
    def __init__(self, max_arity, nfeat, nhid, nemb, alpha, dropout, nheads):
        super().__init__()
        self.emb_dim = nemb
        self.dropout = dropout
        self.attentions = [SparseHyperGraphAttentionLayer(nfeat, nhid, max_arity, alpha, dropout, concat=True)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)
        self.out_attention = SparseHyperGraphAttentionLayer(nhid * nheads, nemb, max_arity, alpha, dropout, concat=False)

    def forward(self, batch_inputs, node_embs, edge_embs, edge_list, node_list):
        node_embs_list, edge_embs_list = [], []
        for attention in self.attentions:
            x, y = attention(node_embs, edge_embs, edge_list, node_list)
            node_embs_list.append(x)
            edge_embs_list.append(y)
        node_embs = torch.cat(node_embs_list, dim=1)
        node_embs = F.dropout(node_embs, self.dropout, training=self.training)
        edge_embs = torch.cat(edge_embs_list, dim=1)
        edge_embs = F.dropout(edge_embs, self.dropout, training=self.training)

        node_embs, edge_embs = self.out_attention(node_embs, edge_embs, edge_list, node_list)
        node_embs, edge_embs = F.elu(node_embs), F.elu(edge_embs)

        batch_outputs = torch.zeros(batch_inputs.shape).unsqueeze(2).repeat_interleave(self.emb_dim, 2)
        batch_outputs[:, 0, :] = edge_embs[batch_inputs[:, 0], :]
        for i in range(len(batch_inputs)):
            right = batch_inputs[i, :-1].nonzero()[0][-1] + 1
            batch_outputs[i, 1:right, :] = node_embs[batch_outputs[i, 1:right], :]
        return batch_outputs
