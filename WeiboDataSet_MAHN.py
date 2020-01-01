from __future__ import print_function
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import math
import random
import queue
import datetime
import re

# trans the original ids to 0~n-1
class IndexDict:
    def __init__(self, original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        cnt = 0
        for i in original_ids:
            new = self.original_to_new.get(i, cnt)  # dict.get(key, default=None)return the value of parameter key
            if new == cnt:
                self.original_to_new[i] = cnt
                cnt += 1
                self.new_to_original.append(i)

    def new(self, original):
        if type(original) is int:
            return self.original_to_new[original]
        else:
            if type(original[0]) is int:
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]

    def original(self, new):
        if type(new) is int:
            return self.new_to_original[new]
        else:
            if type(new[0]) is int:
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]

    def length(self):
        return len(self.new_to_original)


class WeiboDataSet(data.Dataset):
    def __init__(self, cascades, observation_time, n_time_interval, feature_num):
        self.cascades = cascades
        self.observation_time = observation_time
        self.n_time_interval = n_time_interval
        self.feature_num = feature_num
        self.embedding = nn.Embedding(1000, feature_num)

    def __getitem__(self, index):
        line = self.cascades[index]
        line = line.split('\t')
        cascade_id = int(line[0])
        starter_id = int(line[1])
        start_time = int(line[2])
        num_nodes = int(line[3])
        cascade_path = line[4].split()
        label = int(line[-1])
        label = np.log(label + 1.0) / np.log(2.0)
        original_ids = set()        # original identity of users

        for k in range(len(cascade_path)):
            item = cascade_path[k].split(":")
            source = int(item[0])
            target = int(item[1])
            # add origion id into set
            original_ids.add(source)
            original_ids.add(target)

        index = IndexDict(original_ids)
        adj_length = index.length()
        adj_interval = np.zeros((adj_length, adj_length), int)
        # user infected time
        arrive_time_array = np.zeros((adj_length, 1), int)  # n * 1

        temporal_features = np.zeros((adj_length, self.feature_num), int)

        interval_adj = np.zeros((self.n_time_interval, index.length()), int)
        for path in cascade_path:
            item = path.split(":")
            source = index.new(int(item[0]))
            target = index.new(int(item[1]))
            # time string to datetime
            time = int(item[2])
            if time != 0 and time % 3600 == 0:
                time = time - 1
            arrive_time_array[target][0] = time//60
            minutes = time // 60
            temp_interval = minutes - arrive_time_array[source][0]
            if temp_interval > 0:
                adj_interval[source][target] = temp_interval + 1
            else:
                adj_interval[source][target] = 1
                # print("Arriving order wrong!")
            temporal_features[target][minutes] = 1
            temp_period = time // 600
            for y in range(temp_period, self.n_time_interval):
                interval_adj[y][target] = 1

        adj_interval = adj_interval + np.eye(adj_length)
        adj_interval[0][0] = 1
        adj = torch.FloatTensor(adj_interval)
        features = torch.FloatTensor(temporal_features)
        interval_adj = torch.FloatTensor(interval_adj)
        arrive_time_array = torch.LongTensor(arrive_time_array)

        label = torch.FloatTensor(np.array([label]))

        return adj, features, interval_adj, arrive_time_array, label
    
    def __len__(self):
        return len(self.cascades)
