import time
import math
import torch
import argparse
import numpy as np
from torch.autograd import backward
from models import *
from dataset import Dataset

NODE_MAX_ARITY = 6

class Experiment:
    def __init__(self, args):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.neg_ratio = args.nr
        self.test = args.test
        self.epochs = args.epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt = args.opt
        self.weight_decay = args.weight_decay
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.alpha = args.alpha
        self.dropout = args.dropout
        self.nheads = args.nheads
        self.dataset = Dataset(args.dataset, NODE_MAX_ARITY)
        print('relation_num={}, entity_num={}\nmax_arity={}'.format(self.dataset.relation_cnt, self.dataset.entity_cnt, self.dataset.max_arity))

        self.node_embs = torch.FloatTensor(np.random.randn(self.dataset.entity_cnt, self.emb_dim)).to(self.device)
        self.edge_embs = torch.FloatTensor(np.random.randn(self.dataset.relation_cnt, self.emb_dim)).to(self.device)
        self.load_model()
    
    def load_model(self):
        self.model = HyperGAT(self.node_embs, self.edge_embs, self.dataset.max_arity, self.emb_dim, self.hidden_dim, self.emb_dim, self.alpha, self.dropout, self.nheads)
        if self.device != torch.device('cpu'):
            self.model.cuda()
        if self.opt == 'Adagrad':
            self.opt = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt == "Adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # TODO: model ckpt save & load

    def decompose_predictions(self, targets, predictions, max_length):
        positive_indices = np.where(targets > 0)[0]
        seq = []
        for ind, val in enumerate(positive_indices):
            if(ind == len(positive_indices)-1):
                seq.append(self.padd(predictions[val:], max_length))
            else:
                seq.append(self.padd(predictions[val:positive_indices[ind + 1]], max_length))
        return seq

    def padd(self, a, max_length):
        b = F.pad(a, (0, max_length - len(a)), 'constant', -math.inf)
        return b

    def padd_and_decompose(self, targets, predictions, max_length):
        seq = self.decompose_predictions(targets, predictions, max_length)
        return torch.stack(seq)

    def batch_loss(self, batch_data, batch_labels):
        loss_layer = torch.nn.CrossEntropyLoss()
        x = batch_data[:, 0, :]
        for i in range(1, self.dataset.data_arity):
            x = x * batch_data[:, i, :]
        y = torch.sum(x, dim=1)
        # TODO: readout this part
        number_of_positive = len(np.where(batch_labels > 0)[0])
        predictions = self.padd_and_decompose(batch_labels, y, self.neg_ratio*self.dataset.data_arity).to(self.deivce)
        targets = torch.zeros(number_of_positive).long().to(self.device)
        loss = loss_layer(predictions, targets)
        return loss

    def train_and_eval(self):
        print('Training the model...')
        print('Number of training data points: {}'.format(len(self.dataset.data['train'])))

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
                loss = self.batch_loss(batch_outputs, batch_labels)
                loss.backward()
                self.opt.step()
                it_ed = time.time()
                print('Iteration #{}: loss={:.4f}, time={:.4f}'.format(it, loss.item(), it_ed - it_st))
                epoch_loss.append(loss.item())
            
            epoch_ed = time.time()
            print('Epoch #{}: avg_loss={}, epoch_time={}'.format(epoch, sum(epoch_loss) / len(epoch_loss), epoch_ed - epoch_st))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='JF17K')
    parser.add_argument("-test", action="store_true", help="If -test is set, then you must specify a -pretrained model. "
                        + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument('-lr', type=float, default=5e-5)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-alpha', type=float, default=0.2)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-weight_decay', type=float, default=5e-6)
    parser.add_argument('-nheads', type=int, default=3)
    parser.add_argument('-emb_dim', type=int, default=100)
    parser.add_argument('-hidden_dim', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-opt', type=str, default="Adagrad")
    args = parser.parse_args()

    experiment = Experiment(args)

    if args.test:
        pass
    else:
        print("************** START OF TRAINING ********************")
        experiment.train_and_eval()