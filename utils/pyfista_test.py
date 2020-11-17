"""
Utils to create data for tests on synthetic data
"""
import torch
from utils.pyfista import FistaFunction
import numpy as np


def create_random_dictionary(normalize=False):
    """
    Creates a random (normal) dictionary.
    :param normalize: Bool. Normalize L0 norm of dictionary if True.
    :return: Tensor. Created dictionary
    """
    dictionary = torch.rand((64, 512))
    if normalize:
        dictionary = dictionary.__div__(torch.norm(dictionary, p=2, dim=0))

    return dictionary


def create_normalized_noised_inputs(normalize_dictionary, normalize_x, sparsity_num=10, num_samples=1000):
    """
    Creates random data based on random (normal) dictionary.
    The data is created as a random linear combination of the dictionary atoms. A random (normal) noise is added to the data.
    :param normalize_dictionary: Bool. Underlying dictionary is normalized if True.
    :param normalize_x: Bool. Normalize L0 norm of data if True.
    :param sparsity_num: Int. Number of dictionary atoms to use in the creation of the data
    :param num_samples: Int. Number of samples to generate.
    :return: Tensor. Created data
    """


    dictionary = create_random_dictionary(normalize_dictionary)
    X, atoms, coefs = create_sparse_input(dictionary, K=sparsity_num, num_samples=num_samples)
    noise = torch.randn(X.size())
    X += noise

    if normalize_x:
        X = X.__div__(torch.norm(X, p=2, dim=0))

    return X, dictionary


def load_real_inputs(normalize_dictionary, normalize_x, sample_id=5):
    """
    Creates a random dictionary and loads an encoding saved from an untrained encoder.
    :param normalize_dictionary: Bool. Underlying dictionary is normalized if True.
    :param normalize_x: Bool. Normalize L0 norm of data if True.
    :param sample_id: Int. Id of data sample to load
    :return: Tensor, Tensor. Created data, created dictionary
    """

    # Create random dictionary
    dictionary = create_random_dictionary(normalize_dictionary)

    # Load data
    data = torch.from_numpy(np.load('unlearned_encodings/unlearned_cifar10_{}.npy'.format(sample_id))).reshape(-1, 64).t()
    inds = list(np.random.choice(list(range(data.size()[1])), 1000, replace=False))
    X = data[:, inds]

    if normalize_x:
        X = X.__div__(torch.norm(X, p=2, dim=0))

    return X, dictionary


def create_sparse_input(dictionary, K=1, num_samples=1):
    """
    Create sparse data given a dictionary and sparsity value.
    :param dictionary: Tensor. Dictionary to base the data on.
    :param K: Int. Sparsity value, use this number of atoms to create the data.
    :param num_samples: Number of samples to create.
    :return: Tensor. Created data
    """
    atoms = torch.randint(dictionary.size()[1], (num_samples, K))
    coefs = torch.randn((K, num_samples, 1))

    X = []
    for sample_ind in range(num_samples):
        input = dictionary[:, atoms[sample_ind,:]].mm(coefs[:, sample_ind])

        X.append(input)

    X = torch.stack(X, 1).squeeze(-1)

    return X, atoms, coefs