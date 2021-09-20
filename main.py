from dataset import Dataset

ds = Dataset('JF17K', {'node': 6, 'edge': 10})
print('relation_num={}, entity_num={}'.format(ds.relation_cnt, ds.entity_cnt))