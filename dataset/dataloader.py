import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class Dataloader():
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))
        np.random.shuffle(self.indices)
        self.split = int(np.floor(0.1 * self.dataset_size))
        self.train_indices = self.indices[self.split:]
        self.test_indices = self.indices[:self.split]
        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        self.test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.test_sampler)
        self.print_info()

    def print_info(self):
        print("Number of training/test patches:", (len(self.train_indices),len(self.test_indices)))