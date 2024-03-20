import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

# ====================== Default ======================

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, n_iter):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.n_iter = n_iter+1
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.template = None

        self.template = open(os.path.join(root, name.split("/")[0],"template.txt"), "r").read()

    def __getraw__(self, item):
        word, pos1, pos2, mask, pos_mask = self.encoder.tokenize(item['tokens'],
                item['h'][2][0],
                item['t'][2][0],
                template=self.template)
        return word, pos1, pos2, mask, pos_mask

    def __additem__(self, d, word, pos1, pos2, mask, pos_mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['pos_mask'].append(pos_mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask, pos_mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                pos_mask = torch.tensor(pos_mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, pos_mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, pos_mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            pos_mask = torch.tensor(pos_mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask, pos_mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return self.n_iter

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

# ====================== TD ======================

class FewRelPromptDatasetTD(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, n_iter):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.n_iter = n_iter+1
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.template = open(os.path.join(root, name.split("/")[0],"template.txt"), "r").read()
        # self.mapping = json.load(open(os.path.join(root, name.split("/")[0], "mapping.json")))
        self.desc = json.load(open(os.path.join(root, name.split("/")[0], "pid2name.json")))

        # map class labels to prompt tokens
        # self.json_data = {self.mapping[k]:v for k,v in self.json_data.items() if k in self.mapping}
        # self.classes = [self.mapping[c] for c in self.classes if c in self.mapping]
        # self.desc = {self.mapping[k]:v for k,v in self.desc.items() if k in self.mapping}

    def __getraw__(self, item):
        word, pos1, pos2, mask, pos_mask = self.encoder.tokenize(item['tokens'],
                item['h'][2][0],
                item['t'][2][0],
                template=self.template)
        return word, pos1, pos2, mask, pos_mask

    def __additem__(self, d, word, pos1, pos2, mask, pos_mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['pos_mask'].append(pos_mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)

        # tokenize descriptions
        desc_word, desc_mask, desc_pos_mask = self.encoder.tokenize_desc([self.desc[c] for c in target_classes])

        support_set = {'desc_word': desc_word, 'desc_mask':desc_mask, 'desc_pos_mask':desc_pos_mask, 'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': [] }
        query_set = {'desc_word': desc_word, 'desc_mask':desc_mask, 'desc_pos_mask':desc_pos_mask, 'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask, pos_mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                pos_mask = torch.tensor(pos_mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, pos_mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, pos_mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask, pos_mask = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            pos_mask = torch.tensor(pos_mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask, pos_mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return self.n_iter
    
def collate_fn_prompt_td(data):
    batch_support = {'desc_word':[], 'desc_mask':[], "desc_pos_mask":[], 'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': []}
    batch_query = {'desc_word':[], 'desc_mask':[], "desc_pos_mask":[], 'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos_mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label

# ====================== get_loader ======================

def get_loader(name, encoder, N, K, Q, batch_size, n_iter,
        num_workers=0, collate_fn=collate_fn, na_rate=0, root='./data'):
    
    if encoder.desc:
        dataset = FewRelPromptDatasetTD(name, encoder, N, K, Q, na_rate, root, n_iter=n_iter)
        data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn_prompt_td
            )
    else:
        dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, n_iter=n_iter)
        data_loader = data.DataLoader(dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
                collate_fn=collate_fn)
    return data_loader