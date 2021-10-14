import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SparseHyperGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, rel_features, max_arity, alpha, dropout, device, concat=True):
        super().__init__()
        self.device = device
        self.max_arity = max_arity
        self.in_features = in_features
        self.out_features = out_features
        self.rel_features = rel_features
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat = concat
        self.special_spmm = SpecialSpmm()

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(rel_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a1 = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

    def forward(self, entity_embs, relation_embs, tuples, H, node_indices, edge_indices):
        N, M = H.shape[0], H.shape[1]

        entity_w = entity_embs.mm(self.W1)
        relation_w = relation_embs.mm(self.W2)

        zeros = -9e15*torch.ones((tuples.shape[0], tuples.shape[1]-1)).to(self.device)
        a1us = entity_w[tuples[:, 1:]-1, :].matmul(self.a1[:self.out_features]).squeeze()
        edge_4att1 = torch.where(tuples[:, 1:] > 0, a1us, zeros)
        edge_4att2 = relation_w[tuples[:, 0]-1].matmul(self.a1[self.out_features:])
        attention = F.softmax(self.leakyrelu(edge_4att1 + edge_4att2), dim=1).unsqueeze(2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        edge_embs = F.elu(torch.sum(attention * entity_w[tuples[:, 1:]-1, :], dim=1))

        indices = H._indices()
        edge_h = torch.cat((edge_embs[edge_indices, :], entity_w[node_indices, :]), dim=1)
        edge_e = torch.exp(-self.leakyrelu(edge_h.matmul(self.a2).squeeze()))
        assert not torch.isnan(edge_e).any()
        
        e_rowsum = self.special_spmm(indices, edge_e, torch.Size([N, M]), torch.ones(size=(M, 1), device=self.device))
        edge_e = F.dropout(edge_e, self.dropout, training=self.training)

        out_entity_embs = self.special_spmm(indices, edge_e, torch.Size([N, M]), edge_embs)
        assert not torch.isnan(out_entity_embs).any()

        out_entity_embs = out_entity_embs.div(e_rowsum)
        assert not torch.isnan(out_entity_embs).any()

        if self.concat:
            out_entity_embs = F.elu(out_entity_embs)

        return out_entity_embs
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class RankingLayer(nn.Module):
    def __init__(self, data_arity, dropout):
        super().__init__()
        self.data_arity = data_arity
        self.dropout = dropout
    
    def forward(self, x):
        y = x[:, 0, :]
        for i in range(1, self.data_arity):
            y = y * x[:, i, :]
        y = F.dropout(y, self.dropout, training=self.training)
        y = torch.sum(y, dim=1)
        return y
