import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms, utils

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
# from models.pixelsnail import PixelSNAIL
from models.fista_pixelsnail import FistaPixelSNAIL
from scheduler import CycleScheduler

import argparse
import pickle
import os
#h
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
from torchvision import datasets
from dataset import CodeRow, NamedDataset
from models.vqvae import VQVAE
import torch.nn as nn
from utils import util_funcs
from models.model_utils import get_model, get_dataset
from dataset import LMDBDataset
import numpy as np
from tensorboardX import SummaryWriter
import datetime


def train(args, epoch, loader, model, optimizer, scheduler, device, writer, experiment_name, vqvae_model):
    loader = tqdm(loader, desc='PixelSnail training {}'.format(experiment_name))

    criterion = nn.CrossEntropyLoss()
    multilabel_criterion = nn.BCEWithLogitsLoss()
    kl_criterion = nn.KLDivLoss()

    total_coefficients_loss = 0
    total_num_nonzeros_loss = 0
    total_atom_loss = 0
    total_steps = 0
    total_loss = 0
    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            top = top.to(device)
            target = top
            reconstruction, num_nonzeros, sigma_matrix, coefficients = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom

            if hasattr(model, 'prepare_inputs'):  # False if using DataParallel
                used_atoms_mask, gt_num_nonzeros = model.prepare_inputs(bottom)
            else:
                used_atoms_mask, gt_num_nonzeros = model.module.prepare_inputs(bottom)

            sampled_atoms, sampled_num_nonzeros, coefficients = model(bottom, used_atoms_mask, gt_num_nonzeros)

        if i % 25 == 0:
            save_reconstruction(bottom, coefficients, epoch, vqvae_model, i, 'train')

        # Todo: Expose different loss weights as script parameters
        atom_loss = multilabel_criterion(sampled_atoms, used_atoms_mask.float())
        num_nonzeros_loss = criterion(sampled_num_nonzeros, gt_num_nonzeros)
        coefficients_loss = kl_criterion(coefficients, target)
        loss = coefficients_loss
        # loss = atom_loss + num_nonzeros_loss + coefficients_loss

        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # TODO: Plan what we want to log
        total_steps += 1
        total_coefficients_loss += coefficients_loss.item()
        total_num_nonzeros_loss += num_nonzeros_loss.item()
        total_atom_loss += atom_loss.item()
        total_loss += loss.item()

        lr = optimizer.param_groups[0]['lr']

        loader.set_postfix(
            {
                'Epoch': epoch + 1,
                'Loss': f'{loss.item():.5f}',
                'Coefficients loss': f'{coefficients_loss.item():.5f}',
                'Num nonzeros loss': f'{num_nonzeros_loss.item():.5f}',
                'Atom selection loss': f'{atom_loss.item():.5f}',
                'LR': f'{lr:.5f}'
            }
        )

        loader.update(1)

    return total_coefficients_loss / total_steps, total_num_nonzeros_loss / total_steps, total_atom_loss / total_steps, total_loss / total_steps


def save_reconstruction(inthing, out, epoch, vqvae_model, i, phase):
    X1 = vqvae_model.decode_code(out.to(next(vqvae_model.parameters()).device))
    X2 = vqvae_model.decode_code(inthing.clone().detach().to(next(vqvae_model.parameters()).device))
    utils.save_image(
        torch.cat([X1, X2], 0),
        'dumps/fista_pixelsnail_dumps/pixelsnail_reconstrution_epoch[{}]_batch[{}]_phase[{}].png'.format(epoch,i , phase),
        nrow=2,
        normalize=True,
        range=(-1, 1),
    )


def test(args, epoch, loader, model, optimizer, scheduler, device, writer, experiment_name, vqvae_model):
    loader = tqdm(loader, desc='PixelSnail testing {}'.format(experiment_name))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_accuracy = 0
    total_steps = 0
    total_loss = 0
    for i, (top, bottom, label) in enumerate(loader):
        if args.hier == 'top':
            top = top.to(device)
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom)
            # out, _ = model(bottom, condition=top)


        if i % 25 == 0:
            save_reconstruction(bottom, out, epoch, vqvae_model, i, 'train')

        loss = criterion(out, target)

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        total_accuracy += accuracy
        total_steps += 1
        total_loss += loss.item()

        loader.set_postfix(
            {
                'Epoch': epoch + 1,
                'Loss': f'{loss.item():.5f}',
                'Acc': f'{accuracy:.5f}'
            }
        )

        loader.update(1)

    return total_accuracy / total_steps, total_loss / total_steps


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


def create_run(architecture, dataset, num_embeddings, num_workers, selection_fn, neighborhood, device, embed_dim, size, **kwargs):
    global args, scheduler

    # Get VQVAE experiment name
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn=selection_fn, size=size, **kwargs)

    # Prepare logger
    writer = SummaryWriter(os.path.join('runs', 'pixelsnail_' + experiment_name + '2', str(datetime.datetime.now())))

    # Load datasets
    test_loader, train_loader = load_datasets(args, experiment_name, num_workers, dataset)

    # Create model and optimizer
    model, optimizer = prepare_model_parts(train_loader)

    # Get checkpoint path for underlying VQ-VAE model
    checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, kwargs['ckpt_epoch'])
    checkpoint_path = f'checkpoint/{checkpoint_name}'

    # Load underlying VQ-VAE model for logging purposes
    vqvae_model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=False, **kwargs)
    vqvae_model.load_state_dict(torch.load(os.path.join(checkpoint_path)), strict=False)
    vqvae_model = vqvae_model.to(args.device)
    vqvae_model.eval()

    # Train model
    train_coefficients_loss, train_num_nonzeros_loss, train_atom_loss, train_losses, \
    test_coefficients_loss, test_num_nonzeros_loss, test_atom_loss, test_losses, = \
    run_train(args, experiment_name, model, optimizer, scheduler, test_loader, train_loader, writer, vqvae_model)

    return train_coefficients_loss, train_num_nonzeros_loss, train_atom_loss, train_losses, \
           test_coefficients_loss, test_num_nonzeros_loss, test_atom_loss, test_losses


def run_train(args, experiment_name, model, optimizer, scheduler, test_loader, train_loader, writer, vqvae_model):
    train_coefficients_loss = []
    train_num_nonzeros_loss = []
    train_atom_loss = []
    train_losses = []
    test_coefficients_loss = []
    test_num_nonzeros_loss = []
    test_atom_loss = []
    test_losses = []
    for i in range(args.pixelsnail_epoch):
        # Train epoch
        avg_train_coefficients_loss, avg_train_num_nonzeros_loss, avg_train_atom_loss, avg_train_losses = \
        train(args, i, train_loader, model, optimizer, scheduler, args.device, writer, experiment_name, vqvae_model)

        # Test epoch
        avg_test_coefficients_loss, avg_test_num_nonzeros_loss, avg_test_atom_loss, avg_test_losses = \
        test(args, i, train_loader, model, optimizer, scheduler, args.device, writer, experiment_name, vqvae_model)

        # Log train outputs
        train_coefficients_loss.append(avg_train_coefficients_loss)
        train_num_nonzeros_loss.append(avg_train_num_nonzeros_loss)
        train_num_nonzeros_loss.append(avg_train_atom_loss)
        train_atom_loss.append(avg_train_losses)
        train_losses.append(avg_train_losses)
        writer.add_scalar('train/coefficients_loss', avg_train_coefficients_loss)
        writer.add_scalar('train/num_nonzeros_loss', avg_train_num_nonzeros_loss)
        writer.add_scalar('train/atom_loss', avg_train_atom_loss)
        writer.add_scalar('train/loss', avg_train_losses)

        # Log test outputs
        test_coefficients_loss.append(avg_test_coefficients_loss)
        test_num_nonzeros_loss.append(avg_test_num_nonzeros_loss)
        test_num_nonzeros_loss.append(avg_test_atom_loss)
        test_atom_loss.append(avg_test_losses)
        test_losses.append(avg_train_losses)
        writer.add_scalar('test/coefficients_loss', avg_test_coefficients_loss)
        writer.add_scalar('test/num_nonzeros_loss', avg_test_num_nonzeros_loss)
        writer.add_scalar('test/atom_loss', avg_test_atom_loss)
        writer.add_scalar('test/loss', avg_test_losses)

        # Create checkpoint
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoint/pixelsnail_{experiment_name}_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )

    return train_coefficients_loss, train_num_nonzeros_loss, train_atom_loss, train_losses, \
           test_coefficients_loss, test_num_nonzeros_loss, test_atom_loss, test_losses


def prepare_model_parts(train_loader):
    global args, scheduler

    # Load specific checkpoint to continue training
    ckpt = {}
    if args.pixelsnail_ckpt is not None:
        ckpt = torch.load(args.pixelsnail_ckpt)
        args = ckpt['args']

    # Create PixelSnail object
    if args.hier == 'top':
        model = FistaPixelSNAIL(
            [args.size // 8, args.size // 8],
            512,
            args.pixelsnail_channel,
            5,
            4,
            args.pixelsnail_n_res_block,
            args.pixelsnail_n_res_channel,
            dropout=args.pixelsnail_dropout,
            n_out_res_block=args.pixelsnail_n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = FistaPixelSNAIL(
            [args.size // 4, args.size // 4],
            512,
            args.pixelsnail_channel,
            5,
            4,
            args.pixelsnail_n_res_block,
            args.pixelsnail_n_res_channel,
            attention=False,
            dropout=args.pixelsnail_dropout,
            n_cond_res_block=args.pixelsnail_n_cond_res_block,
            cond_res_channel=args.pixelsnail_n_res_channel,
        )

    # Load saved checkpoint into PixelSnail object
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    # Parallelize training
    model = nn.DataParallel(model)

    # Move model to proper device
    model = model.to(args.device)

    # Create other training objects
    optimizer = optim.Adam(model.parameters(), lr=args.pixelsnail_lr)
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    scheduler = None
    if args.pixelsnail_sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.pixelsnail_lr, n_iter=len(train_loader) * args.pixelsnail_epoch, momentum=None
        )
    return model, optimizer


def load_datasets(args, experiment_name, num_workers, dataset):
    """
    Load LMDB datasets
    """
    db_name = util_funcs.create_checkpoint_name(experiment_name, args.ckpt_epoch)[:-3] + '_dataset[{}]'.format(dataset)

    train_dataset = LMDBDataset(os.path.join('codes', 'train_codes', db_name), args.architecture)
    test_dataset = LMDBDataset(os.path.join('codes', 'test_codes', db_name ), args.architecture)

    train_loader = DataLoader(train_dataset, batch_size=args.pixelsnail_batch, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.pixelsnail_batch, shuffle=True, num_workers=num_workers)
    return test_loader, train_loader


def log_arguments(**arguments):
    experiment_name = util_funcs.create_experiment_name(**arguments)
    with open(os.path.join('checkpoint', experiment_name + '_args.txt'), 'w') as f:
        for key in arguments.keys():
            f.write('{} : {} \n'.format(key, arguments[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    parser = util_funcs.pixelsnail_parser(parser)
    args = parser.parse_args()

    print(args)
    log_arguments(**vars(args))
    create_run(**vars(args))
