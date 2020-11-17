import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class NamedDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return list(self.dataset[index]) + [index]

    def __len__(self):
        return len(self.dataset)


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path, architecture):
        if architecture == 'vqvae' or architecture == 'vqvae2':
            self.architecture = architecture
        else:
            raise ValueError('Valid architectures are vqvae and vqvae2. Got: {}'.format(architecture))


        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        if self.architecture == 'vqvae':
            return torch.from_numpy(row.bottom), torch.from_numpy(row.bottom), row.filename
        elif self.architecture == 'vqvae2':
            return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename