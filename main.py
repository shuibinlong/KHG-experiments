import argparse
from dataset import Dataset

class Experiment:
    def __init__(self, args):
        self.lr = args.lr
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.neg_ratio = args.nr
        self.test = args.test
        self.epochs = args.epochs

        self.max_arity = {'node': args.node, 'edge': args.edge}
        self.dataset = Dataset(args.dataset, self.max_arity)
        print('relation_num={}, entity_num={}'.format(self.dataset.relation_cnt, self.dataset.entity_cnt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    experiment = Experiment(args)