import os
import numpy as np
import torch


class Dataset:
    def __init__(self, ds_name, data_arity):
        self.data_arity = data_arity
        self.max_arity = {'edge': 0, 'node': 0}
        self.entity2id, self.entity_cnt = {}, 0
        self.relation2id, self.relation_cnt = {}, 0
        self.edge_list, self.node_list = {}, {}
        self.data = {}
        self.batch_index = 0

        print(f'Loading the dataset {ds_name} ...')
        dir = os.path.join('data', ds_name)
        self.data['train'] = self.read_train(os.path.join(dir, 'train.txt'))
        np.random.shuffle(self.data['train'])
        self.data['test'] = self.read_test(os.path.join(dir, 'test.txt'))
        if ds_name == 'JF17K':
            for i in range(2, self.data_arity):
                test_arity = f'test_{i}'
                self.data[test_arity] = self.read_test(os.path.join(dir, f'{test_arity}.txt'))
        self.data['valid'] = self.read_train(os.path.join(dir, 'valid.txt'))
        self.process_list()

    def read_train(self, file_path):
        if not os.path.exists(file_path):
            print(f'[ERROR] {file_path} not found, skipping')
            return {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data = np.zeros((len(lines), self.data_arity + 1))
        for i, line in enumerate(lines):
            record = line.strip().split('\t')
            data[i] = self.record2ids(record)
            self.parse_adj(record)
        return data

    def read_test(self, file_path):
        if not os.path.exists(file_path):
            print(f'[ERROR] {file_path} not found, skipping')
            return {}
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data = np.zeros((len(lines), self.data_arity + 1))
        for i, line in enumerate(lines):
            record = line.strip().split('\t')
            data[i] = self.record2ids(record[1:])
            self.parse_adj(record[1:])
        return data
    
    def process_list(self):
        for i in range(1, self.relation_cnt + 1):
            raw = np.array(list(self.edge_list[i]))
            fixed = np.zeros(self.max_arity['node'] - len(self.edge_list[i]))
            self.edge_list[i] = np.concatenate((raw, fixed))
            assert self.edge_list[i].shape[0] == self.max_arity['node']
        for i in range(1, self.entity_cnt + 1):
            raw = np.array(list(self.node_list[i]))
            fixed = np.zeros(self.max_arity['edge'] - len(self.node_list[i]))
            self.node_list[i] = np.concatenate((raw, fixed))
            assert self.node_list[i].shape[0] == self.max_arity['edge']

    def parse_adj(self, record):
        edge = self.get_relation_id(record[0])
        for x in record[1:]:
            node = self.get_entity_id(x)
            self.insert_edge(edge, node)
            self.insert_node(node, edge)  

    def insert_edge(self, edge, node):
        if edge not in self.edge_list:
            self.edge_list[edge] = set()
        self.edge_list[edge].add(node)
        self.max_arity['node'] = max(self.max_arity['node'], len(self.edge_list[edge]))
    
    def insert_node(self, node, edge):
        if node not in self.node_list:
            self.node_list[node] = set()
        self.node_list[node].add(edge)
        self.max_arity['edge'] = max(self.max_arity['edge'], len(self.node_list[node]))

    def record2ids(self, record):
        res = np.zeros(self.data_arity + 1)
        for i, x in enumerate(record):
            res[i] = self.get_relation_id(x) if i == 0 else self.get_entity_id(x)
        return res

    def get_relation_id(self, x):
        if x not in self.relation2id:
            self.relation_cnt += 1
            self.relation2id[x] = self.relation_cnt
        return self.relation2id[x]

    def get_entity_id(self, x):
        if x not in self.entity2id:
            self.entity_cnt += 1
            self.entity2id[x] = self.entity_cnt
        return self.entity2id[x]

    def get_next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.get_pos_batch(batch_size)
        batch = self.gen_neg_batch(pos_batch, neg_ratio)
        batch_edges = torch.Tensor(batch[:, :-2]).long().to(device)
        batch_labels = torch.Tensor(batch[:, -2]).long().to(device)
        return batch_edges, batch_labels

    def get_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data['train']):
            batch = self.data['train'][self.batch_index: self.batch_index + batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data['train'][self.batch_index:]
            np.random.shuffle(self.data['train'])
            self.batch_index = 0
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype('int')  # label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype('int')  # arity
        return batch

    def gen_neg_batch(self, pos_batch, neg_ratio):
        arities = [(x != 0).sum() for x in pos_batch]
        pos_batch[:, -1] = arities
        neg_batch = np.concatenate(
            [self.gen_neg(np.repeat([x], neg_ratio * arities[i] + 1, axis=0), arities[i], neg_ratio) for i, x in
             enumerate(pos_batch)], axis=0)
        return neg_batch

    def gen_neg(self, neg, arity, batch):
        neg[0, -2] = 1
        for i in range(arity):
            neg[i * batch + 1: (i + 1) * batch + 1, i + 1] = np.random.randint(1, self.entity_cnt + 1, size=batch)
        return neg
