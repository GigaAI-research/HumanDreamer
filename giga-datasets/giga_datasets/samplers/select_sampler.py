import math

import numpy as np
import torch


class SelectSampler(torch.utils.data.Sampler):
    # 选择数据集中特定idx的数据
    def __init__(self, dataset, batch_size=None, shuffle=True, infinite=True, dataset_name=None,seed=6666):
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size = len(dataset)
        if batch_size is not None:
            self.total_size = int(math.ceil(self.data_size / batch_size)) * batch_size
        else:
            self.total_size = self.data_size
        self.dataset = dataset
        self.dataset_name = dataset_name
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            if self.dataset_name == 'dance_all':
                indices = [1119,1745,3439,4526,1092,676]
            elif self.dataset_name == 'k400':
                indices = [3310, 158, 3452, 4038, 912]
            else:
                raise ValueError(f'Unknown dataset name: {self.dataset_name}')
            # indices = [1119,1745,3439,4526,1092,676,1318,2186...]

            # indices = [4800, 496, 2317, 3650, 3327, 5788, 1086, 3310, 158, 3452, 3680, 2442, 5591, 4038, 537, 701, 4404, 5911, 912, 2304]
            # indices = [3310, 158, 3452, 4038, 912]  # k400
            # indices = np.zeros((0,), dtype=np.int64)
            # while len(indices) < self.total_size:
            #     indices_i = np.arange(self.data_size)
            #     if self.shuffle:
            #         indices_i = np.random.permutation(indices_i)
            #     num_data = min(len(indices_i), self.total_size - len(indices))
            #     indices = np.hstack((indices, indices_i[:num_data]))
            yield from indices
            if not self.infinite:
                break
