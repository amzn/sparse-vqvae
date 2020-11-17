"""
Goal of this script is to visualize the encodings created using extract_code.py for debugging purposes
"""
import argparse
# import pickle
import os

import torch
from torch.utils.data import DataLoader
# from torchvision import transforms
# import lmdb
# from tqdm import tqdm
# from torchvision import datasets
# from dataset import CodeRow, NamedDataset
# from models.vqvae import VQVAE
# import torch.nn as nn
from utils import util_funcs
from models.model_utils import get_model, get_dataset
from torchvision import datasets, transforms, utils
# import joblib
from dataset import LMDBDataset


def create_run(architecture, dataset, num_embeddings, num_workers, selection_fn, neighborhood, device, size, ckpt_epoch, embed_dim, **kwargs):
    global args, scheduler

    print('creating data loaders')
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood,
                                                        selection_fn, size, **kwargs)
    checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)
    checkpoint_path = f'checkpoint/{checkpoint_name}'

    test_loader, train_loader = load_datasets(args, experiment_name, num_workers, dataset)

    print('Loading model')
    model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=False, **kwargs)
    model.load_state_dict(torch.load(os.path.join('..', checkpoint_path)), strict=False)
    model = model.to(device)
    model.eval()

    for batch in train_loader:
        print('decoding')
        X = model.decode_code(batch[1].to(next(model.parameters()).device))

        print('decoded')
        utils.save_image(
            torch.cat([X], 0),
            'X_img.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        a = 5
        exit()


def load_datasets(args, experiment_name, num_workers, dataset):
    db_name = util_funcs.create_checkpoint_name(experiment_name, args.ckpt_epoch)[:-3] + '_dataset[{}]'.format(dataset)
    train_dataset = LMDBDataset(os.path.join('..', 'codes', 'train_codes', db_name), args.architecture)
    test_dataset = LMDBDataset(os.path.join('..', 'codes', 'test_codes', db_name), args.architecture)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers, drop_last=True)
    return test_loader, train_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    args = parser.parse_args()

    print(args)

    util_funcs.seed_generators(args.seed)

    create_run(**vars(args))
