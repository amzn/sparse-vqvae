import argparse
import os
import shutil

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from models.vqvae import VQVAE2, VQVAE
from models.pixelsnail import PixelSNAIL
from models.model_utils import get_model

from utils import util_funcs

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0]), desc='Sampling rows'):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample.detach()

    return row


def load_model(model, checkpoint, device, architecture=None, num_embeddings=None, neighborhood=None, selection_fn=None, size=256, **kwargs):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, **kwargs)

    elif model == 'vqvae2':
        model = VQVAE2()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [size//8, size//8],
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
            [size//4, size//4],
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


def create_run(device, temp, pixelsnail_batch, ckpt_epoch, pixelsnail_ckpt_epoch, hier, architecture, num_embeddings, neighborhood, selection_fn, dataset, size, **kwargs):
    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn, size, **kwargs)
    vqvae_checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)

    # pixelsnail_checkpoint_name = f'pixelsnail_{experiment_name}_{hier}_{str(pixelsnail_ckpt_epoch + 1).zfill(3)}.pt'
    pixelsnail_checkpoint_name = 'pixelsnail_vqvae_imagenet_num_embeddings[512]_neighborhood[1]_selectionFN[vanilla]_size[128]_bottom_420.pt'

    # model_vqvae = load_model('vqvae', vqvae_checkpoint_name, device, architecture, num_embeddings, neighborhood, selection_fn, **kwargs)
    # model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', pixelsnail_checkpoint_name, device, size=size, **kwargs)

    num_samples = 50000
    sampled_directory = os.path.join('sampled_images', pixelsnail_checkpoint_name).replace('.pt', '')
    if os.path.exists(sampled_directory):
        shutil.rmtree(sampled_directory)
    os.mkdir(sampled_directory)

    for sample_ind in tqdm(range(num_samples), 'Sampling image for: {}'.format(pixelsnail_checkpoint_name)):

        # top_sample = sample_model(model_top, device, args.batch, [32, 32], args.temp)
        bottom_sample = sample_model(
            model_bottom, device, pixelsnail_batch, [size // 4, size // 4], temp, condition=None
            # model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
        )

        # decoded_sample = model_vqvae._modules['module'].decode_code(bottom_sample)
        # decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
        # decoded_sample = decoded_sample.clamp(-1, 1)

        # filename = 'sampled_{}.png'.format(sample_ind)
        # target_path = os.path.join(sampled_directory, filename)
        save_image(decoded_sample, target_path, normalize=True, range=(-1, 1))



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

