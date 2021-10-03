import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseHyperGraphAttentionLayer

class HyperGAT(nn.Module):
    def __init__(self, node_embs, edge_embs, max_arity, nfeat, nhid, nemb, alpha, dropout, nheads, device):
        super().__init__()
        self.emb_dim = nemb
        self.dropout = dropout
        self.device = device
        self.node_embs = nn.Parameter(node_embs)
        self.edge_embs = nn.Parameter(edge_embs)
       
        self.attentions = [SparseHyperGraphAttentionLayer(nfeat, nhid, max_arity, alpha, dropout, device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)
        self.out_attention = SparseHyperGraphAttentionLayer(nhid * nheads, nemb, max_arity, alpha, dropout, device, concat=False)

    def forward(self, batch_inputs, edge_list, node_list):
        node_embs_list, edge_embs_list = [], []
        for attention in self.attentions:
            x, y = attention(self.node_embs, self.edge_embs, edge_list, node_list)
            node_embs_list.append(x)
            edge_embs_list.append(y)
        out_node_embs = torch.cat(node_embs_list, dim=1)
        out_node_embs = F.dropout(out_node_embs, self.dropout, training=self.training)
        out_edge_embs = torch.cat(edge_embs_list, dim=1)
        out_edge_embs = F.dropout(out_edge_embs, self.dropout, training=self.training)

        out_node_embs, out_edge_embs = self.out_attention(out_node_embs, out_edge_embs, edge_list, node_list)
        out_node_embs, out_edge_embs = F.elu(out_node_embs), F.elu(out_edge_embs)

        batch_outputs = torch.ones(batch_inputs.shape).unsqueeze(2).repeat_interleave(self.emb_dim, 2)
        batch_outputs[:, 0, :] = out_edge_embs[batch_inputs[:, 0]-1, :]
        for i in range(len(batch_inputs)):
            right = batch_inputs[i, :].nonzero()[-1][0] + 1
            batch_outputs[i, 1:right, :] = out_node_embs[batch_inputs[i, 1:right]-1, :]
        return batch_outputs
