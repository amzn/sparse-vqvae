import argparse
import os
import shutil

import threading
import logging
import time

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.vqvae import VQVAE2, VQVAE
from models.pixelsnail import PixelSNAIL
from models.model_utils import get_model

from utils import util_funcs


@torch.no_grad()
def sample_model(thread_id, model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    # for i in range(size[0]):
    for i in tqdm(range(size[0]), desc='Thread {}, sampling rows'.format(thread_id)):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device, architecture=None, num_embeddings=None, neighborhood=None, selection_fn=None,
               **kwargs):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, **kwargs)

    elif model == 'vqvae2':
        model = VQVAE2()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [args.size//8, args.size//8],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [args.size//4, args.size//4],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    model.eval()

    return model


def sample_from_range(thread_ind, min_ind, max_ind, sampled_directory, device, temp, batch, ckpt_epoch, pixelsnail_ckpt_epoch, hier, architecture, num_embeddings, neighborhood, selection_fn, size, dataset, **kwargs):
    logging.info("Sampling thread {}: starting with range [{},{}) on device {}".format(thread_ind, min_ind, max_ind, device))
    pixelsnail_checkpoint_name, vqvae_checkpoint_name = get_checkpoint_names(architecture, ckpt_epoch, dataset, hier,
                                                                             kwargs, neighborhood, num_embeddings,
                                                                             pixelsnail_ckpt_epoch, selection_fn, size)

    model_bottom, model_vqvae = load_models(architecture, device, kwargs, neighborhood, num_embeddings, pixelsnail_checkpoint_name, selection_fn, vqvae_checkpoint_name)
    # print('Sampling in range {}-{}'.format(min_ind, max_ind))
    # for sample_ind in tqdm(range(min_ind, max_ind), 'Sampling image for: {}'.format(pixelsnail_checkpoint_name)):
    for sample_ind in tqdm(range(min_ind, max_ind), 'Sampling image for: {} in range [{},{})'.format(pixelsnail_checkpoint_name, min_ind, max_ind)):
        logging.info('Thread {}, sample ind {}'.format(thread_ind, sample_ind))
        bottom_sample = sample_model(thread_ind, model_bottom, device, batch, [size//4, size//4], temp, condition=None)

        decoded_sample = model_vqvae._modules['module'].decode_code(bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)

        filename = 'sampled_{}.png'.format(sample_ind)
        target_path = os.path.join(sampled_directory, filename)
        save_image(decoded_sample, target_path, normalize=True, range=(-1, 1))


def load_models(architecture, device, kwargs, neighborhood, num_embeddings, pixelsnail_checkpoint_name, selection_fn,
                vqvae_checkpoint_name):
    model_vqvae = load_model('vqvae', vqvae_checkpoint_name, device, architecture, num_embeddings, neighborhood,
                             selection_fn, **kwargs)
    model_bottom = load_model('pixelsnail_bottom', pixelsnail_checkpoint_name, device, **kwargs)
    return model_bottom, model_vqvae


def get_checkpoint_names(architecture, ckpt_epoch, dataset, hier, kwargs, neighborhood, num_embeddings,
                         pixelsnail_ckpt_epoch, selection_fn, size):
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn, size,
                                                        **kwargs)
    vqvae_checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)
    pixelsnail_checkpoint_name = f'pixelsnail_{experiment_name}_{hier}_{str(pixelsnail_ckpt_epoch + 1).zfill(3)}.pt'
    return pixelsnail_checkpoint_name, vqvae_checkpoint_name


def create_run(device, temp, batch, ckpt_epoch, pixelsnail_ckpt_epoch, hier, architecture, num_embeddings, neighborhood,
               selection_fn, dataset, num_threads, size, **kwargs):
    pixelsnail_checkpoint_name, _ = get_checkpoint_names(architecture, ckpt_epoch, dataset, hier,
                                                                             kwargs, neighborhood, num_embeddings,
                                                                             pixelsnail_ckpt_epoch, selection_fn, size)
    sampled_directory = os.path.join('sampled_images', pixelsnail_checkpoint_name).replace('.pt', '')
    if os.path.exists(sampled_directory):
        shutil.rmtree(sampled_directory)
    os.mkdir(sampled_directory)

    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    threads = list()
    num_samples = 50000
    min_ind = 0
    step_size = num_samples // num_threads
    for thread_index in range(num_threads):
        max_ind = min_ind + step_size
        x = threading.Thread(target=sample_from_range, args=(thread_index, min_ind, max_ind, sampled_directory, device + ':{}'.format(thread_index), temp, batch, ckpt_epoch, pixelsnail_ckpt_epoch, hier, architecture, num_embeddings, neighborhood, selection_fn, size, dataset), kwargs=kwargs)
        threads.append(x)
        x.start()
        min_ind = max_ind
        if thread_index + 1 == num_threads:
            min_ind = max(min_ind, num_samples - step_size)

    for thread_index, thread in enumerate(threads):
        thread.join()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    parser = util_funcs.pixelsnail_parser(parser)
    parser = util_funcs.sampling_parser(parser)
    args = parser.parse_args()

    print(args)

    create_run(**vars(args))

