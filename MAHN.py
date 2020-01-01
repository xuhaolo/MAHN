import numpy as np
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


class MAHNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, observation_time, concat=True, outlayer=True):
        super(MAHNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.observation_time = observation_time
        self.concat = concat
        self.outlayer = outlayer

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features), gain=np.sqrt(2.0)), requires_grad=True)
        self.decay_weight1 = nn.Parameter(torch.abs(torch.normal(mean=torch.zeros((60*observation_time, 1)), std=0.02)), requires_grad=True)
        self.decay_weight2 = nn.Parameter(torch.abs(torch.normal(mean=torch.zeros((60*observation_time, 1)), std=0.02)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj, arrive_time_array, observation_time):
        h = torch.mm(input, self.W)
        sparse_adj = adj.to_sparse()
        oh_interval = torch.zeros(sparse_adj.values().size()[0], 60*observation_time, device='cuda').scatter(1, (sparse_adj.values()-1).long().reshape(-1, 1), 1)
        decay_list = torch.mm(oh_interval, self.decay_weight1).squeeze()
        h = F.relu(h)
        h_prime = self.special_spmm(sparse_adj.indices(), decay_list, sparse_adj.shape, h)
        if self.outlayer:
            windows_interval = 60 * observation_time - arrive_time_array - 1
            windows_interval = windows_interval.reshape(-1, 1)
            onehot_windows_interval = torch.zeros(windows_interval.size()[0], observation_time*60, device='cuda').scatter_(1, windows_interval, 1)

            windows_decay = torch.mm(onehot_windows_interval, self.decay_weight2)
            h_prime = windows_decay * h_prime
            
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MAHN_1L(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, observation_time):
        """Dense version of GAT."""
        super(MAHN_1L, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.out_att = [MAHNLayer(nfeat, nclass, dropout=dropout, alpha=alpha, observation_time=observation_time, concat=True, outlayer=True) for _ in range(nheads)]
        for i, att in enumerate(self.out_att):
            self.add_module('att_{}'.format(i), att)
    
    def forward(self, x, adj, arrive_time_array, observation_time):
        # x = F.dropout(x, self.dropout, training=self.training)
        tmp = [att(x, adj, arrive_time_array, observation_time) for att in self.out_att]
        output = 0
        for t in tmp:
            output += t
        output = output / self.nheads
        return output


class MAHN_2L(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, observation_time):
        """Dense version of GAT."""
        super(MAHN_2L, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.attentions = [MAHNLayer(nfeat, nhid, dropout=dropout, alpha=alpha, observation_time=observation_time, concat=True, outlayer=False) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = [MAHNLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, observation_time=observation_time, concat=True, outlayer=True) for _ in range(nheads)]
        for i, att in enumerate(self.out_att):
            self.add_module('att_{}'.format(i), att)

    def forward(self, x, adj, arrive_time_array, observation_time):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, arrive_time_array, observation_time) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        tmp = [att(x, adj, arrive_time_array, observation_time) for att in self.out_att]
        output = 0
        for t in tmp:
            output += t
        output = output / self.nheads
        return output


