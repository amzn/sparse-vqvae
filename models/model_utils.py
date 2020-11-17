import os
# import torch
import torch.nn as nn
from models.vqvae import VQVAE
from torchvision import datasets, transforms, utils


def get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=True,  **kwargs):
    """
    Creates a VQVAE object.

    :param architecture: Has to be 'vqvae'.
    :param num_embeddings: Int. Number of dictioanry atoms
    :param device: String. 'cpu', 'cuda' or 'cuda:device_number'
    :param neighborhood: Int. Not used.
    :param selection_fn: String. 'vanilla' or 'fista'
    :param embed_dim: Int. Size of latent space.
    :param parallel: Bool. Use DataParallel or not.

    :return: VQVAE model or DataParallel(VQVAE model)
    """
    if architecture == 'vqvae':
        model = VQVAE(n_embed=num_embeddings, neighborhood=neighborhood, selection_fn=selection_fn, embed_dim=embed_dim, **kwargs).to(device)
    else:
        raise ValueError('Valid architectures are vqvae. Got: {}'.format(architecture))

    if parallel and device != 'cpu':
        model = nn.DataParallel(model)

    return model


def get_dataset(dataset, data_path, size, download=False):
    """
    Loads a dataset

    :param dataset: String. Name of dataset. Currently supports ['test', 'cifar10', 'cifar100', 'imagenet'].
    Note that 'test' loads the 'cifar10'
    :param data_path: String. Path to directory where the dataset is / will be saved.
    :param size: Int. Resize image to this size.
    :param download: Bool. True to download dataset IF needed. False will save time.
    :return: Dataset object.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if dataset == 'test':
        train_dataset = datasets.CIFAR10(root=os.path.join(data_path, dataset), download=download,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(root=os.path.join(data_path, dataset), download=download,
                                        transform=test_transform, train=False)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=os.path.join(data_path, dataset), download=download,
                                         transform=train_transform)
        test_dataset = datasets.CIFAR10(root=os.path.join(data_path, dataset), download=download,
                                        transform=test_transform, train=False)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=os.path.join(data_path, dataset), download=download,
                                          transform=train_transform)
        test_dataset = datasets.CIFAR100(root=os.path.join(data_path, dataset), download=download,
                                         transform=test_transform, train=False)
    elif dataset == 'imagenet':
        train_dataset = datasets.ImageNet(root=os.path.join(data_path, dataset), download=download,
                                          transform=train_transform, split='train')
        test_dataset = datasets.ImageNet(root=os.path.join(data_path, dataset), download=False, # Currently the ImageNet valiadation set is inaccessible
                                         transform=test_transform, split='val')
    else:
        raise ValueError('Valid datasets are cifar10, cifar100 and imagenet. Got: {}'.format(dataset))

    return train_dataset, test_dataset