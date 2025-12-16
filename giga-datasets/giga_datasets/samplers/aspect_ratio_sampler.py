import math
import copy
import numpy as np
import torch

from ..datasets import ConcatDataset, Dataset


class AspectRatioSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        aspect_ratios,
        batch_size,
        height_name='image_height',
        width_name='image_width',
        min_size=1,
        shuffle=True,
        infinite=True,
        seed=6666,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.indices_list = _get_indices_list(
            dataset=dataset,
            aspect_ratios=aspect_ratios,
            height_name=height_name,
            width_name=width_name,
            min_size=min_size,
        )
        self.total_size_list = []
        for indices in self.indices_list:
            data_size = len(indices)
            if batch_size is not None:
                total_size = int(math.ceil(data_size / batch_size)) * batch_size
            else:
                total_size = data_size
            self.total_size_list.append(total_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return sum(self.total_size_list)

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            indices_list = []
            for i in range(len(self.indices_list)):
                indices = np.zeros((0,), dtype=np.int64)
                total_size = self.total_size_list[i]
                while len(indices) < total_size:
                    indices_i = self.indices_list[i]
                    if self.shuffle:
                        indices_i = np.random.permutation(indices_i)
                    num_data = min(len(indices_i), total_size - len(indices))
                    indices = np.hstack((indices, indices_i[:num_data]))
                if self.batch_size is not None:
                    indices = indices.reshape((-1, self.batch_size))
                indices_list.append(indices)
            indices = np.concatenate(indices_list, axis=0)
            if self.shuffle:
                indices = np.random.permutation(indices)
            indices = indices.reshape(-1)
            yield from indices
            if not self.infinite:
                break


def _process_dataset(dataset):
    if isinstance(dataset, ConcatDataset):
        dataset.datasets = [_process_dataset(_) for _ in dataset.datasets]
    elif isinstance(dataset, Dataset):
        dataset.datasets = [dataset.datasets[0]]
    dataset.transform = None
    return dataset


def _get_indices_list(dataset, aspect_ratios, height_name, width_name, min_size):
    dataset = _process_dataset(copy.deepcopy(dataset))
    data_size = len(dataset)
    aspect_ratios = np.array(aspect_ratios)
    indices_list = [[] for _ in range(len(aspect_ratios))]
    for i in range(data_size):
        data_dict = dataset[i]
        height = data_dict[height_name]
        width = data_dict[width_name]
        aspect_ratio = float(height) / width
        idx = np.argmin(np.abs(aspect_ratios - aspect_ratio))
        indices_list[idx].append(i)
    new_indices_list = []
    for indices in indices_list:
        if len(indices) >= min_size:
            indices = np.array(indices, dtype=np.int64)
            new_indices_list.append(indices)
    return new_indices_list
