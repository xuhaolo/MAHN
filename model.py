import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MAHN


# model with 1 layer
class MAHN_1L(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_time_interval, observation_time):
        super(MAHN_1L, self).__init__()
        self.nfeat = nfeat
        self.alpha = alpha
        self.n_time_interval = n_time_interval
        self.MAHN = MAHN.MAHN_1L(nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha,
                observation_time=observation_time)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # Conv1d
        self.output = nn.Sequential(
            nn.Conv1d(1, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj, interval_adj, arrive_time_array, observation_time):
        x = self.MAHN(x, adj, arrive_time_array, observation_time)
        x = x.sum(dim=0)
        x = x.view(1, 1, -1)
        x = self.output(x)
        x = x.view(1, -1)
        x = F.relu(x)
        return x


# model with 2 layers
class MAHN_2L(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_time_interval, observation_time):
        super(MAHN_2L, self).__init__()
        self.nfeat = nfeat
        self.alpha = alpha
        self.n_time_interval = n_time_interval
        self.MAHN = MAHN.MAHN_2L(nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nheads=nheads,
                alpha=alpha,
                observation_time=observation_time)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        # Conv1d
        self.output = nn.Sequential(
            nn.Conv1d(1, 100, 65),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Conv1d(100, 1, 64),
        )

    def forward(self, x, adj, interval_adj, arrive_time_array, observation_time):
        x = self.MAHN(x, adj, arrive_time_array, observation_time)
        x = x.sum(dim=0)
        x = x.view(1, 1, -1)
        x = self.output(x)
        x = x.view(1, -1)
        x = F.relu(x)
        return x




