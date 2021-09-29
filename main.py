import time
import torch
import argparse
import numpy as np
from models import *
from dataset import Dataset

class Experiment:
    def __init__(self, args):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.neg_ratio = args.nr
        self.test = args.test
        self.epochs = args.epochs
        self.device = args.device
        self.opt = args.opt
        self.weight_decay = args.weight_decay
        self.max_arity = {'node': args.node, 'edge': args.edge}
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.nheads = args.nheads
        self.dataset = Dataset(args.dataset, self.max_arity)
        print('relation_num={}, entity_num={}'.format(self.dataset.relation_cnt, self.dataset.entity_cnt))

        self.node_embs = torch.FloatTensor(np.random.randn(self.dataset.entity_cnt, self.emb_dim)).to(args.device)
        self.edge_embs = torch.FloatTensor(np.random.randn(self.dataset.relation_cnt, self.emb_dim)).to(args.device)
    
    def load_model(self):
        self.model = HyperGAT(self.node_embs, self.edge_embs, self.max_arity, self.emb_dim, self.hidden_dim, self.emb_dim, self.alpha, self.dropout, self.nheads)
        if self.opt == 'Adagrad':
            self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # TODO: model ckpt save & load

    def train_and_eval(self):
        print('Training the model...')
        print('Number of training data points: {}'.format(len(self.dataset.data['train'])))

        loss_layer = torch.nn.CrossEntropyLoss()
        print('Starting training at iteration ...')
        for epoch in range(self.epochs):
            self.model.train()
            epoch_st = time.time()
            epoch_loss = []
            
            num_iterations = len(self.dataset.data['train']) // self.batch_size
            if len(self.dataset.data['train']) % self.batch_size != 0:
                num_iterations += 1
            
            for it in range(num_iterations):
                it_st = time.time()
                batch_data, batch_labels = self.dataset.get_next_batch(self.batch_size, self.neg_ratio, self.device)
                batch_outputs = self.model(batch_data, self.dataset.edge_list, self.dataset.node_list)
                # TODO: calc loss & bp
                self.opt.step()
                it_ed = time.time()
                print('Iteration #{}: loss={:.4f}, time={:.4f}'.format(it, 0, it_ed - it_st))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    experiment = Experiment(args)