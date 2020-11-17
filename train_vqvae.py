import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from scheduler import CycleScheduler
from tensorboardX import SummaryWriter

from utils.util_funcs import *
from models.model_utils import get_model, get_dataset
from datetime import datetime
import numpy as np


_TODAY = datetime.now().strftime("%Y_%m_%d")


def do_epoch(epoch_num, loader, model, writer, experiment_name, device, optimizer=None, scheduler=None, phase='train',
             dictionary_loss_weight=0, sampling_iter=50, sample_size=25):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    _today = getattr(args, 'today_str', _TODAY)
    tqdm_loader = tqdm(loader, desc='{}: {}'.format(phase, experiment_name))

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0
    loss_sum = 0
    total_steps = 0
    quantization_steps = 0
    avg_dictionary_embedding_size = 0
    avg_Z_size = 0
    avg_norm_Z = 0
    avg_top_percentile = 0
    avg_num_zeros = 0

    for i, (img, label) in enumerate(tqdm_loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros = model(img)
        recon_loss = criterion(out, img)
        encoder_latent_loss = latent_loss[0].mean()
        dictionary_latent_loss = latent_loss[1].mean()
        loss = recon_loss + latent_loss_weight * encoder_latent_loss + dictionary_latent_loss * dictionary_loss_weight

        if phase == 'train':
            loss.backward()

            if scheduler is not None:
                scheduler.step()
            optimizer.step()

        mse_sum += recon_loss.item()
        loss_sum += encoder_latent_loss.item()
        total_steps += 1
        quantization_steps += num_quantization_steps.mean().item()
        avg_dictionary_embedding_size += mean_D.mean().item()
        avg_Z_size += mean_Z.mean().item()
        avg_norm_Z += norm_Z.mean().item()
        avg_top_percentile += top_percentile.mean().item()
        avg_num_zeros += num_zeros.mean().item()

        lr = optimizer.param_groups[0]['lr']

        tqdm_loader.set_postfix({
            # 'Experiment': experiment_name,
            'Epoch': epoch_num,
            'mse': recon_loss.item(),
            'latent_loss': encoder_latent_loss.item(),
            # 'avg_norm_Z': norm_Z.mean().item(),
        })
        tqdm_loader.update(1)

        # writer.add_scalar('{}/epoch_{}/latent_loss'.format(phase, epoch_num), encoder_latent_loss.item(), i)
        # writer.add_scalar('{}/epoch_{}/avg_mse'.format(phase, epoch_num), recon_loss.item(), i)
        # writer.add_scalar('{}/epoch_{}/norm_Z'.format(phase, epoch_num), norm_Z.mean(), i)
        # writer.add_scalar('{}/epoch_{}/top_percentile'.format(phase, epoch_num), top_percentile.mean(), i)
        # writer.add_scalar('{}/epoch_{}/num_zeros'.format(phase, epoch_num), num_zeros.mean(), i)

        if i % sampling_iter == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out = model(sample)
                out = out[0]

            sample_path = os.path.join(args.summary_path, _today, experiment_name, 'samples')
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            utils.save_image(
                torch.cat([sample, out], 0), os.path.join(sample_path, f'Epoch_{str(epoch_num).zfill(5)}_batch_{str(i).zfill(5)}.png'),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            if phase == 'train':
                model.train()

    return mse_sum / total_steps, loss_sum / total_steps, quantization_steps / total_steps, avg_Z_size / total_steps, avg_dictionary_embedding_size / total_steps, avg_norm_Z / total_steps, avg_top_percentile / total_steps, avg_num_zeros / total_steps


def create_training_run(size, num_epochs, lr, sched, dataset, architecture, data_path, device, num_embeddings, neighborhood, selection_fn, num_workers, vae_batch, eval_iter, embed_dim, parallelize, download, **kwargs):
    experiment_name = create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn=selection_fn, size=size, lr=lr, **kwargs)
    _today = getattr(args, 'today_str', _TODAY)
    log_arguments(**vars(args))
    writer = SummaryWriter(os.path.join(args.summary_path, _today, experiment_name))

    print('Loading datasets')
    train_dataset, test_dataset = get_dataset(dataset, data_path, size, download)

    print('Creating loaders')
    train_loader = DataLoader(train_dataset, batch_size=vae_batch, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=vae_batch, shuffle=True, num_workers=num_workers)

    print('Initializing models')
    model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallelize, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, lr, n_iter=len(train_loader) * num_epochs, momentum=None
        )

    train_mses = []
    train_losses = []
    test_mses = []
    test_losses = []

    for epoch_ind in range(1, num_epochs+1):
        avg_mse, avg_loss, avg_quantization_steps, avg_Z, avg_D, avg_norm_Z, avg_top_percentile, avg_num_zeros = do_epoch(epoch_ind, train_loader, model, writer, experiment_name, device, optimizer, scheduler, dictionary_loss_weight=kwargs['dictionary_loss_weight'], sampling_iter=kwargs['sampling_iter'],sample_size=kwargs['sample_size'])
        train_mses.append(avg_mse)
        train_losses.append(avg_loss)
        writer.add_scalar('train/avg_mse', avg_mse, epoch_ind)
        writer.add_scalar('train/avg_loss', avg_loss, epoch_ind)
        writer.add_scalar('train/avg_quantization_steps', avg_quantization_steps, epoch_ind)
        writer.add_scalar('train/avg_Z', avg_Z, epoch_ind)
        writer.add_scalar('train/avg_D', avg_D, epoch_ind)
        writer.add_scalar('train/avg_norm_Z', avg_norm_Z, epoch_ind)
        writer.add_scalar('train/avg_top_percentile', avg_top_percentile, epoch_ind)
        writer.add_scalar('train/avg_num_zeros', avg_num_zeros, epoch_ind)

        if epoch_ind % kwargs['checkpoint_freq'] == 0:

            cp_path = os.path.join(args.checkpoint_path, _today, create_checkpoint_name(experiment_name, epoch_ind))
            os.makedirs(osp.dirname(cp_path), exist_ok=True)
            if parallelize:  # If using DataParallel we need to access the inner module
                torch.save(model.module.state_dict(), cp_path)
            else:
                torch.save(model.state_dict(), cp_path)

        if epoch_ind % eval_iter == 0:
            avg_mse, avg_loss, avg_quantization_steps, avg_Z, avg_D, avg_norm_Z, avg_top_percentile, avg_num_zeros = do_epoch(epoch_ind, test_loader, model, writer, experiment_name, device, optimizer, scheduler, phase='test')

            test_mses.append(avg_mse)
            test_losses.append(avg_loss)
            writer.add_scalar('test/avg_loss', avg_loss, epoch_ind)
            writer.add_scalar('test/avg_mse', avg_mse, epoch_ind)
            writer.add_scalar('test/avg_quantization_steps', avg_quantization_steps, epoch_ind)
            writer.add_scalar('test/avg_Z', avg_Z, epoch_ind)
            writer.add_scalar('test/avg_D', avg_D, epoch_ind)
            writer.add_scalar('test/avg_norm_Z', avg_norm_Z, epoch_ind)
            writer.add_scalar('test/avg_top_percentile', avg_top_percentile, epoch_ind)
            writer.add_scalar('test/avg_num_zeros', avg_num_zeros, epoch_ind)
            model.train()

    return train_mses, train_losses, test_mses, test_losses


def log_arguments(**arguments):
    _today = arguments.get('today_str', _TODAY)
    experiment_name = create_experiment_name(**arguments)
    cp_path = os.path.join(arguments['checkpoint_path'], _today, experiment_name)
    os.makedirs(cp_path, exist_ok=True)
    with open(os.path.join(cp_path, 'args.txt'), 'w') as f:
        for key in arguments.keys():
            f.write('{} : {} \n'.format(key, arguments[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = base_parser(parser)
    parser = vqvae_parser(parser)
    parser = code_extraction_parser(parser)
    args = parser.parse_args()

    print(str(args).replace(',', ',\n\t'))  #[print(f'\t{k}: {v}') for k,v in arguments.items()]
    args.today_str = datetime.now().strftime('%Y_%m_%d')
    seed_generators(args.seed)
    device = args.device

    create_training_run(**vars(args))
