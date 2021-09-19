import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseHyperGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, max_arity, alpha, dropout):
        super().__init__()
        self.max_arity = max_arity
        self.in_features = in_features
        self.out_features = out_features
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout

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
            if a1u.shape[0] < self.max_arity['node']:
                zero = -9e15*torch.ones(size=(self.max_arity['node']-a1u.shape[0], 1))
                a1u = torch.cat((a1u, zero), dim=0)
            return a1u
        
        attention = F.softmax(torch.stack([get_a1u(i) for i in range(M)], dim=0), dim=1)
        
        def get_Whs(i):
            Whs = Wh[edge_list[i], :]
            if Whs.shape[0] < self.max_arity['node']:
                zero = torch.zeros(size=(self.max_arity['node']-Whs.shape[0], self.out_features))
                Whs = torch.cat((Whs, zero), dim=0)
            return Whs

        new_edge_embs = torch.sum(attention * torch.stack([get_Whs(i) for i in range(M)], dim=0), dim=1)
        assert new_edge_embs.shape == edge_embs.shape
        new_edge_embs = F.dropout(new_edge_embs, self.dropout, training=self.training)

        # node attention
        Wf = new_edge_embs.mm(self.W2)
        
        def get_a2v(i):
            a2v = Wf[node_list[i], :]
            hi_repeat = node_embs[i].expand(a2v.shape)
            a2v = torch.cat((a2v, hi_repeat), dim=1).mm(self.a2)
            if a2v.shape[0] < self.max_arity['edge']:
                zero = -9e15*torch.ones(size=(self.max_arity['edge']-a2v.shape[0], 1))
                a2v = torch.cat((a2v, zero), dim=0)
            return a2v
        
        attention = F.softmax(torch.stack([get_a2v(i) for i in range(N)], dim=0), dim=1)

        def get_Wfs(i):
            Wfs = Wf[node_list[i], :]
            if Wfs.shape[0] < self.max_arity['edge']:
                zero = torch.zeros(size=(self.max_arity['edge']-Wfs.shape[0], self.out_features))
                Wfs = torch.cat((Wfs, zero), dim=0)
            return Wfs

        new_node_embs = torch.sum(attention * torch.stack([get_Wfs(i) for i in range(N)], dim=0), dim=1)
        assert new_node_embs.shape == node_embs.shape
        new_node_embs = F.dropout(new_node_embs, self.dropout, training=self.training)

        return new_node_embs, new_edge_embs