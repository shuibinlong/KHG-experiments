import torch
from measure import Measure

class Tester:
    def __init__(self, model, dataset, valid_or_test, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples())
    
    def test(self, test_by_arity):
        normalizer = 0
        self.measure_by_arity = {}
        self.meaddsure = Measure()
        if test_by_arity:
            for cur_arity in range(2, self.dataset.data_arity + 1):
                test_by_arity = "test_{}".format(cur_arity)
                if len(self.dataset.data.get(test_by_arity, ())) == 0 :
                    print("%%%%% {} does not exist. Skipping.".format(test_by_arity))
                    continue
                print("**** Evaluating arity {} having {} samples".format(cur_arity, len(self.dataset.data[test_by_arity])))
                current_measure, normalizer_by_arity =  self.eval_dataset(self.dataset.data[test_by_arity])
                normalizer += normalizer_by_arity
                self.measure += current_measure
                current_measure.normalize(normalizer_by_arity)
                self.measure_by_arity[test_by_arity] = current_measure
        else:
            current_measure, normalizer =  self.eval_dataset(self.dataset.data[self.valid_or_test])
            self.measure = current_measure
        if normalizer == 0:
            raise Exception("No Samples were evaluated! Check your test or validation data!!")
        self.measure.normalize(normalizer)
        self.measure_by_arity["ALL"] = self.measure
        pr_txt = "Results for ALL ARITIES in {} set".format(self.valid_or_test)
        if test_by_arity:
            for arity in self.measure_by_arity:
                if arity == "ALL":
                    print(pr_txt)
                else:
                    print("Results for arity {}".format(arity[5:]))
                print(self.measure_by_arity[arity])
        else:
            print(pr_txt)
            print(self.measure)
        return self.measure, self.measure_by_arity


    def eval_dataset(self, dataset):
        settings = ["raw", "fil"]
        normalizer = 0
        current_rank = Measure()
        for i, fact in enumerate(dataset):
            arity = self.dataset.data_arity - (fact == 0).sum()
            for j in range(1, arity + 1):
                normalizer += 1
                queries = self.create_queries(fact, j)
                for raw_or_fil in settings:
                    batch_queries = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    sim_scores = self.model(batch_queries, self.dataset.edge_list, self.dataset.node_list).cpu().numpy()
                    rank = self.get_rank(sim_scores)
                    current_rank.update(rank, raw_or_fil)
            if i % 1000 == 0:
                print(f"--- Testing sample {i}")
        return current_rank, normalizer

    def get_rank(self, sim_scores):
        # Assumes the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()
    
    def create_queries(self, fact, position):
        r, e1, e2, e3, e4, e5, e6 = fact

        if position == 1:
            return [(r, i, e2, e3, e4, e5, e6) for i in range(1, self.dataset.entity_cnt + 1)]
        elif position == 2:
            return [(r, e1, i, e3, e4, e5, e6) for i in range(1, self.dataset.entity_cnt + 1)]
        elif position == 3:
            return [(r, e1, e2, i, e4, e5, e6) for i in range(1, self.dataset.entity_cnt + 1)]
        elif position == 4:
            return [(r, e1, e2, e3, i, e5, e6) for i in range(1, self.dataset.entity_cnt + 1)]
        elif position == 5:
            return [(r, e1, e2, e3, e4, i, e6) for i in range(1, self.dataset.entity_cnt + 1)]
        elif position == 6:
            return [(r, e1, e2, e3, e4, e5, i) for i in range(1, self.dataset.entity_cnt + 1)]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == 'raw':
            result = [tuple(fact)] + queries
        elif raw_or_fil == 'fil':
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return self.shred_facts(result)
    
    def shred_facts(self, tuples):
        return torch.LongTensor(tuples).to(self.device)
    
    def allFactsAsTuples(self):
        tuples = []
        for spl in self.dataset.data:
            for fact in self.dataset.data.get(spl, ()):
                tuples.append(tuple(fact))
        return tuples
        