import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
# from torchvision import transforms
import lmdb
from tqdm import tqdm
# from torchvision import datasets
from dataset import CodeRow, NamedDataset
# from models.vqvae import VQVAE
# import torch.nn as nn
from utils import util_funcs
from models.model_utils import get_model, get_dataset
# from torchvision import datasets, transforms, utils
# import joblib


def extract(lmdb_env, loader, model, device, phase='train'):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader, desc='Extracting for {} phase'.format(phase))

        for img, _, filename in pbar:
            img = img.to(device)

            # Quantize the image and output the atom ids of the patches
            quant, _, id, _, _, _, _, _, _ = model.encode(img)
            id = id.detach().cpu().numpy()

            # Dump every patch separately
            for file, bottom in zip(filename, id):
                row = CodeRow(top=None, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_postfix({'Inserted': index})

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


def create_extraction_run(size, device, dataset, data_path, num_workers, num_embeddings, architecture, ckpt_epoch, neighborhood, selection_fn, embed_dim, **kwargs):
    train_dataset, test_dataset = get_dataset(dataset, data_path, size)

    print('Creating named datasets')
    # We don't really use the "Named" part, but I'm keeping it to stay close to the original code repository
    train_named_dataset = NamedDataset(train_dataset)
    test_named_dataset = NamedDataset(test_dataset)

    print('creating data loaders')
    train_loader = DataLoader(train_named_dataset, batch_size=kwargs['vae_batch'], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_named_dataset, batch_size=kwargs['vae_batch'], shuffle=False, num_workers=num_workers)

    # This is still the VQ-VAE experiment name and path
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn, size, **kwargs)
    checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)
    checkpoint_path = f'checkpoint/{checkpoint_name}'

    print('Loading model')
    model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=False, **kwargs)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model = model.to(device)
    model.eval()


    print('Creating LMDB DBs')
    map_size = 100 * 1024 * 1024 * 1024  # This would be the maximum size of the databases
    db_name = checkpoint_name[:-3] + '_dataset[{}]'.format(dataset)  # This comprises of the experiment name and the epoch the codes are taken from
    train_env = lmdb.open(os.path.join('codes', 'train_codes', db_name), map_size=map_size)  # Will save the encodings of train samples
    test_env = lmdb.open(os.path.join('codes', 'test_codes', db_name), map_size=map_size)  # Will save the encodings of test samples

    print('Extracting')
    if architecture == 'vqvae':
        extract(train_env, train_loader, model, device, 'train')
        extract(test_env, test_loader, model, device, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    args = parser.parse_args()

    print(args)

    util_funcs.seed_generators(args.seed)

    create_extraction_run(**vars(args))
    # create_extraction_run(args.size, args.device, args.dataset, args.data_path, args.num_workers, args.num_embed, args.architecture, args.ckpt_epoch, args.neighborhood, args.selection_fn)
