'''
**************************************************************************
The models here are designed to work with datasets having maximum arity 6.
**************************************************************************
'''


import numpy as np
import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
import math

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)


class MDistMult(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MDistMult, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs['hidden_drop']
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data[1:])

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)

        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class MCP(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MCP, self).__init__()
        self.emb_dim = emb_dim
        self.E1 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E2 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E3 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E4 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E5 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.E6 = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.hidden_drop_rate = kwargs['hidden_drop']
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E1.weight.data[0] = torch.ones(self.emb_dim)
        self.E2.weight.data[0] = torch.ones(self.emb_dim)
        self.E3.weight.data[0] = torch.ones(self.emb_dim)
        self.E4.weight.data[0] = torch.ones(self.emb_dim)
        self.E5.weight.data[0] = torch.ones(self.emb_dim)
        self.E6.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E1.weight.data[1:])
        xavier_normal_(self.E2.weight.data[1:])
        xavier_normal_(self.E3.weight.data[1:])
        xavier_normal_(self.E4.weight.data[1:])
        xavier_normal_(self.E5.weight.data[1:])
        xavier_normal_(self.E6.weight.data[1:])
        xavier_normal_(self.R.weight.data)

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E1(e1_idx)
        e2 = self.E2(e2_idx)
        e3 = self.E3(e3_idx)
        e4 = self.E4(e4_idx)
        e5 = self.E5(e5_idx)
        e6 = self.E6(e6_idx)
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class HSimplE(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(HSimplE, self).__init__()
        self.emb_dim = emb_dim
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.hidden_drop_rate = kwargs['hidden_drop']
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R.weight.data)


    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.shift(self.E(e2_idx), int(1 * self.emb_dim/self.max_arity))
        e3 = self.shift(self.E(e3_idx), int(2 * self.emb_dim/self.max_arity))
        e4 = self.shift(self.E(e4_idx), int(3 * self.emb_dim/self.max_arity))
        e5 = self.shift(self.E(e5_idx), int(4 * self.emb_dim/self.max_arity))
        e6 = self.shift(self.E(e6_idx), int(5 * self.emb_dim/self.max_arity))
        x = r * e1 * e2 * e3 * e4 * e5 * e6
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class HypE(BaseClass):
    def __init__(self, d, emb_dim, **kwargs):
        super(HypE, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.filt_h = kwargs['filt_h']
        self.filt_w = kwargs['filt_w']
        self.stride = kwargs['stride']
        self.hidden_drop_rate = kwargs['hidden_drop']
        self.emb_dim = emb_dim
        self.max_arity = 6
        rel_emb_dim = emb_dim
        self.E = torch.nn.Embedding(d.num_ent(), emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(d.num_rel(), rel_emb_dim, padding_idx=0)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, emb_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(rel_emb_dim + self.max_arity + 1, fc1_length)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length)


    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve(self, r_idx, e_idx, pos):

        e = self.E(e_idx).view(-1, 1, 1, self.E.weight.size(1))
        r = self.R(r_idx)
        x = e
        x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):
        r = self.R(r_idx)
        e1 = self.convolve(r_idx, e1_idx, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
        e2 = self.convolve(r_idx, e2_idx, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
        e3 = self.convolve(r_idx, e3_idx, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
        e4 = self.convolve(r_idx, e4_idx, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
        e5 = self.convolve(r_idx, e5_idx, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
        e6 = self.convolve(r_idx, e6_idx, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

        x = e1 * e2 * e3 * e4 * e5 * e6 * r
        x = self.hidden_drop(x)
        x = torch.sum(x, dim=1)
        return x

class MTransH(BaseClass):
    def __init__(self, dataset, emb_dim, **kwargs):
        super(MTransH, self).__init__()
        self.emb_dim = emb_dim
        self.E = torch.nn.Embedding(dataset.num_ent(), emb_dim, padding_idx=0)
        self.R1 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)
        self.R2 = torch.nn.Embedding(dataset.num_rel(), emb_dim, padding_idx=0)

        self.b0 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b1 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b2 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b3 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b4 = torch.nn.Embedding(dataset.num_rel(), 1)
        self.b5 = torch.nn.Embedding(dataset.num_rel(), 1)

        self.hidden_drop_rate = kwargs['hidden_drop']
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)

    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R1.weight.data[0] = torch.ones(self.emb_dim)
        self.R2.weight.data[0] = torch.ones(self.emb_dim)
        xavier_normal_(self.E.weight.data[1:])
        xavier_normal_(self.R1.weight.data[1:])
        xavier_normal_(self.R2.weight.data[1:])
        normalize_entity_emb = F.normalize(self.E.weight.data[1:], p=2, dim=1)
        normalize_relation_emb = F.normalize(self.R1.weight.data[1:], p=2, dim=1)
        normalize_norm_emb = F.normalize(self.R2.weight.data[1:], p=2, dim=1)
        self.E.weight.data[1:] = normalize_entity_emb
        self.R1.weight.data[1:] = normalize_relation_emb
        self.R2.weight.data[1:] = normalize_norm_emb
        xavier_normal_(self.b0.weight.data)
        xavier_normal_(self.b1.weight.data)
        xavier_normal_(self.b2.weight.data)
        xavier_normal_(self.b3.weight.data)
        xavier_normal_(self.b4.weight.data)
        xavier_normal_(self.b5.weight.data)

    def pnr(self, e_idx, r_idx):
        original = self.E(e_idx)
        norm = self.R2(r_idx)
        return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms):
        r = self.R1(r_idx)
        e1 = self.pnr(e1_idx, r_idx) * self.b0(r_idx)
        e1 = e1 * ms[:,0].unsqueeze(-1).expand_as(e1)
        e2 = self.pnr(e2_idx, r_idx) * self.b1(r_idx)
        e2 = e2 * ms[:,1].unsqueeze(-1).expand_as(e2)
        e3 = self.pnr(e3_idx, r_idx) * self.b2(r_idx)
        e3 = e3 * ms[:,2].unsqueeze(-1).expand_as(e3)
        e4 = self.pnr(e4_idx, r_idx) * self.b3(r_idx)
        e4 = e4 * ms[:,3].unsqueeze(-1).expand_as(e4)
        e5 = self.pnr(e5_idx, r_idx) * self.b4(r_idx)
        e5 = e5 * ms[:,4].unsqueeze(-1).expand_as(e5)
        e6 = self.pnr(e6_idx, r_idx) * self.b5(r_idx)
        e6 = e6 * ms[:,5].unsqueeze(-1).expand_as(e6)
        x = r + e1 + e2 + e3 + e4 + e5 + e6
        x = self.hidden_drop(x)
        x = -1 * torch.norm(x, p=2, dim=1)
        return x

class HyperConvR(BaseClass):
    def __init__(self, dataset, device, **kwargs):
        super().__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.device = device
        self.kernel_size = kwargs['conv']['kernel_size']
        self.conv_out_channels = kwargs['conv']['filters']
        self.stride = kwargs['conv']['stride']
        self.emb_dim = {
            'entity': kwargs['entity_emb_dim'],
            'relation': self.kernel_size[0] * self.kernel_size[1] * self.conv_out_channels,
            'reshape': kwargs['conv']['reshape']
        }
        assert self.emb_dim['entity'] == self.emb_dim['reshape'][0] * self.emb_dim['reshape'][1]
        self.feature_map_drop = torch.nn.Dropout2d(kwargs['conv']['feature_map_dropout'])
        self.input_drop = torch.nn.Dropout(kwargs['input_drop'])
        self.hidden_drop = torch.nn.Dropout(kwargs['hidden_drop'])
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim['entity'], padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), self.emb_dim['relation'] * self.max_arity, padding_idx=0)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity'])
        self.bn3 = torch.nn.BatchNorm1d(self.emb_dim['relation'])
        self.filter_h = (self.emb_dim['reshape'][0] - self.kernel_size[0]) // self.stride + 1
        self.filter_w = (self.emb_dim['reshape'][1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * self.filter_h * self.filter_w
        self.fc = torch.nn.Linear(fc_length, self.emb_dim['entity'])
    
    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim['entity'])
        self.R.weight.data[0] = torch.ones(self.emb_dim['relation'] * self.max_arity)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve(self, e_idx, r):
        batch_size = e_idx.shape[0]

        e = self.E(e_idx)
        e = self.bn2(e)
        e = e.view(1, -1, *self.emb_dim['reshape'])
        e = self.input_drop(e)

        r = self.bn3(r)
        kernel = r.view(-1, 1, *self.kernel_size)
        kernel = self.input_drop(kernel)

        x = F.conv2d(e, kernel, stride=self.stride, groups=batch_size)
        x = x.view(batch_size, self.conv_out_channels, self.filter_h, self.filter_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1) # (batch_size, hidden_size)
        x = self.fc(x) # (batch_size, ent_emb_dim)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
    
    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms):
        r = self.R(r_idx).view(-1, self.emb_dim['relation'], self.max_arity)
        e1 = self.convolve(e1_idx, r[:, :, 0]) * ms[:,0].view(-1, 1)
        e2 = self.convolve(e2_idx, r[:, :, 1]) * ms[:,1].view(-1, 1)
        e3 = self.convolve(e3_idx, r[:, :, 2]) * ms[:,2].view(-1, 1)
        e4 = self.convolve(e4_idx, r[:, :, 3]) * ms[:,3].view(-1, 1)
        e5 = self.convolve(e5_idx, r[:, :, 4]) * ms[:,4].view(-1, 1)
        e6 = self.convolve(e6_idx, r[:, :, 5]) * ms[:,5].view(-1, 1)

        y = e1 + e2 + e3 + e4 + e5 + e6
        y = F.relu(y)
        y = self.hidden_drop(y)
        y = torch.sum(y, dim=1)
        return y

class HyperConvE(BaseClass):
    def __init__(self, dataset, device, **kwargs):
        super().__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.device = device
        self.emb_dim = {'entity': kwargs['entity_emb_dim'], 'relation': kwargs['relation_emb_dim'], 'reshape': kwargs['conv']['reshape']}
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim['entity'], padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), self.emb_dim['relation'] * self.max_arity, padding_idx=0)
        self.input_drop = torch.nn.Dropout(kwargs['input_drop'])
        self.hidden_drop = torch.nn.Dropout(kwargs['hidden_drop'])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs['conv']['feature_map_dropout'])
        self.conv_out_channels = kwargs['conv']['filters']
        self.kernel_size = kwargs['conv']['kernel_size']
        self.stride = kwargs['conv']['stride']
        self.conv = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, self.stride, 0, bias=kwargs['conv']['use_bias'])
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity'])
        self.filter_h = (self.emb_dim['reshape'][0] * 2 - self.kernel_size[0]) // self.stride + 1
        self.filter_w = (self.emb_dim['reshape'][1] - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * self.filter_h * self.filter_w
        self.fc = torch.nn.Linear(fc_length, self.emb_dim['entity'])
    
    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim['entity'])
        self.R.weight.data[0] = torch.ones(self.emb_dim['relation'] * self.max_arity)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def convolve(self, e_idx, r):
        batch_size = e_idx.shape[0]
        e = self.E(e_idx).view(-1, 1, self.emb_dim['reshape'][0], self.emb_dim['reshape'][1])
        r = r.view(-1, 1, self.emb_dim['reshape'][0], self.emb_dim['reshape'][1])

        stacked_inputs = torch.cat([e, r], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.input_drop(stacked_inputs)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x        

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx, ms, bs):
        r = self.R(r_idx).view(-1, self.max_arity, self.emb_dim['relation'])
        e1 = self.convolve(e1_idx, r[:, 0, :]) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
        e2 = self.convolve(e2_idx, r[:, 1, :]) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
        e3 = self.convolve(e3_idx, r[:, 2, :]) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
        e4 = self.convolve(e4_idx, r[:, 3, :]) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
        e5 = self.convolve(e5_idx, r[:, 4, :]) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
        e6 = self.convolve(e6_idx, r[:, 5, :]) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)
        
        y = e1 * e2 * e3 * e4 * e5 * e6
        y = self.hidden_drop(y)
        y = torch.sum(y, dim=1)
        return y

class HyperConvKB(BaseClass):
    def __init__(self, dataset, device, **kwargs):
        super().__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(1, dtype=torch.int32), requires_grad=False)
        self.device = device
        self.emb_dim = {'entity': kwargs['entity_emb_dim'], 'relation': kwargs['relation_emb_dim']}
        self.max_arity = 6
        self.E = torch.nn.Embedding(dataset.num_ent(), self.emb_dim['entity'], padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel(), self.emb_dim['relation'], padding_idx=0)
        self.input_drop = torch.nn.Dropout(kwargs['input_drop'])
        self.hidden_drop = torch.nn.Dropout(kwargs['hidden_drop'])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs['conv']['feature_map_dropout'])
        self.conv_out_channels = kwargs['conv']['filters']
        self.kernel_size = kwargs['conv']['kernel_size']
        self.stride = kwargs['conv']['stride']
        self.conv = torch.nn.Conv2d(1, self.conv_out_channels, self.kernel_size, self.stride)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity'])
        self.filter_h = (self.emb_dim['entity'] - self.kernel_size[0]) // self.stride + 1
        self.filter_w = (self.max_arity + 1 - self.kernel_size[1]) // self.stride + 1
        fc_length = self.conv_out_channels * self.filter_h * self.filter_w
        self.fc = torch.nn.Linear(fc_length, 1, bias=False)
    
    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim['entity'])
        self.R.weight.data[0] = torch.ones(self.emb_dim['relation'])
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:])

    def forward(self, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, e6_idx):
        batch_size = r_idx.shape[0]
        
        r = self.R(r_idx)
        e1 = self.E(e1_idx)
        e2 = self.E(e2_idx)
        e3 = self.E(e3_idx)
        e4 = self.E(e4_idx)
        e5 = self.E(e5_idx)
        e6 = self.E(e6_idx)
        
        x = torch.stack([r, e1, e2, e3, e4, e5, e6], dim=2).unsqueeze(1)
        x = self.input_drop(x)
        x = self.bn0(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.hidden_drop(x)
        x = self.fc(x).squeeze()
        return x
