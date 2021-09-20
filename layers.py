import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseHyperGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, max_arity, alpha, dropout, concat=True):
        super().__init__()
        self.max_arity = max_arity
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def forward(self, node_embs, edge_embs, edge_list, node_list):
        N, M = node_embs.shape[0], edge_embs.shape[0]

        # edge attention        
        Wh = node_embs.mm(self.W1)

        def get_a1u(i):
            a1u = self.leakyrelu(Wh[edge_list[i], :]).mm(self.a1)
            zero = -9e15*torch.ones(size=(self.max_arity['node'], 1))
            return torch.where(a1u > 0, a1u, zero)
        
        attention = F.softmax(torch.stack([get_a1u(i) for i in range(M)], dim=0), dim=1)

        new_edge_embs = torch.sum(attention * torch.stack([Wh[edge_list[i], :] for i in range(M)], dim=0), dim=1)
        assert new_edge_embs.shape == edge_embs.shape
        new_edge_embs = F.dropout(new_edge_embs, self.dropout, training=self.training)

        # node attention
        Wf = new_edge_embs.mm(self.W2)

        def get_a2v(i):
            a2v = Wf[node_list[i], :].mm(self.a2[:self.out_features, :])
            a2hi = torch.mm(node_embs[i].unsqueeze(0), self.a2[self.out_features:, :])
            a2v[a2v.nonzero().t()[0], :] += a2hi
            return a2v

        attention = F.softmax(torch.stack([get_a2v(i) for i in range(N)], dim=0), dim=1)

        new_node_embs = torch.sum(attention * torch.stack([Wf[node_list[i], :] for i in range(N)], dim=0), dim=1)
        assert new_node_embs.shape == node_embs.shape
        new_node_embs = F.dropout(new_node_embs, self.dropout, training=self.training)

        if self.concat:
            return F.elu(new_node_embs), F.elu(new_edge_embs)

        return new_node_embs, new_edge_embs
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'