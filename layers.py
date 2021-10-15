import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseHyperGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, max_arity, alpha, dropout, device, concat=True):
        super().__init__()
        self.device = device
        self.max_arity = max_arity
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def forward(self, node_embs, edge_embs, edge_list, node_list):
        N, M = node_embs.shape[0], edge_embs.shape[0]

        # edge attention
        Wh = node_embs.mm(self.W1)

        zeros = -9e15*torch.ones((edge_list.shape[0], edge_list.shape[1])).to(self.device)
        a1us = self.leakyrelu(Wh[edge_list-1]).matmul(self.a1).squeeze()
        edge_4att = torch.where(edge_list > 0, a1us, zeros)
        attention = F.softmax(edge_4att, dim=1).unsqueeze(2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        new_edge_embs = torch.sum(attention * Wh[edge_list-1], dim=1)
        # assert new_edge_embs.shape == edge_embs.shape

        # node attention
        Wf = edge_embs.mm(self.W2)

        zeros = -9e15*torch.ones((node_list.shape[0], node_list.shape[1])).to(self.device)
        a2vs = Wf[node_list-1].matmul(self.a2[:self.out_features]).squeeze()
        node_4att1 = torch.where(node_list > 0, a2vs, zeros)
        node_4att2 = Wh.matmul(self.a2[self.out_features:, :])
        attention = F.softmax(self.leakyrelu(node_4att1 + node_4att2), dim=1).unsqueeze(2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        new_node_embs = torch.sum(attention * Wf[node_list-1], dim=1)
        # assert new_node_embs.shape == node_embs.shape

        if self.concat:
            return F.elu(new_node_embs), F.elu(new_edge_embs)

        return new_node_embs, new_edge_embs
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class RankingLayer(nn.Module):
    def __init__(self, data_arity, dropout):
        super().__init__()
        self.data_arity = data_arity
        self.dropout = dropout
    
    def forward(self, x):
        y = x[:, 0, :]
        for i in range(1, self.data_arity + 1):
            y = y * x[:, i, :]
        y = F.dropout(y, self.dropout, training=self.training)
        y = torch.sum(y, dim=1)
        return y
