import math

import numpy as np
import torch


class TrainTestSampler(torch.utils.data.Sampler):
    """
    给定一个总数据集，根据输入的百分比，逐渐控制部分数据加入训练集。
    例子：做scaling law数据集实验时，分别以20%,40%,60%,80%的数据量进行训练。最后0.1%用于测试。"""
    def __init__(self, dataset, batch_size=None, shuffle=True, infinite=True, seed=6666, stage=None, percentage=1):
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size = len(dataset)
        self.stage = stage
        self.percentage = percentage


        if batch_size is not None:
            self.total_size = int(math.ceil(self.data_size * self.percentage / batch_size)) * batch_size
        else:
            self.total_size = int(self.data_size * self.percentage)


    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            np.random.seed(self.seed+ self.epoch)
            self.epoch += 1
            indices = np.arange(self.data_size)
            
            # 创建一个具有指定种子的新随机数生成器实例
            local_rng = np.random.default_rng(0)

            if self.shuffle:
                # indices = np.random.permutation(indices)
                # 使用局部随机数生成器进行随机排列
                indices = local_rng.permutation(indices)
            if self.stage == 'train':
                # Select only the percentage of data required for training
                train_indices = indices[:int(self.data_size * self.percentage)]
                train_indices = np.random.permutation(train_indices)
                yield from train_indices
            elif self.stage == 'test':
                # Calculate the index where to split for test data
                test_start_index = max(0, self.data_size - int(self.data_size * self.percentage))
                # Get the last X% of the data for testing
                test_indices = indices[test_start_index:]
                yield from test_indices
            else:
                raise ValueError("Stage must be either 'train' or 'test'")
            if not self.infinite:
                break
            # indices = np.zeros((0,), dtype=np.int64)
            # while len(indices) < self.total_size:
            #     indices_i = np.arange(self.data_size)
            #     if self.shuffle:
            #         indices_i = np.random.permutation(indices_i)
            #     if self.stage == 'train':
            #         # Select only the percentage of data required for training
            #         indices = indices_i[:int(self.data_size * self.train_percentage)]
            #     elif self.stage == 'test':
            #         # Calculate the index where to split for test data
            #         test_start_index = max(0, self.data_size - int(self.data_size * self.test_percentage))
            #         # Get the last X% of the data for testing
            #         indices = indices_i[test_start_index:]
                
                # num_data = min(len(indices_i), self.total_size - len(indices))
                # indices = np.hstack((indices, indices_i[:num_data]))
            # yield from indices
            # if not self.infinite:
            #     break
