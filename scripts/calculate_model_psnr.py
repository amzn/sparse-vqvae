"""
Calculates PSNR for a given VQ-VAE model with respect to a dataset.
Accepts same arguments as train_vqvae.py
"""
import sys
import os
sys.path.append(os.path.abspath('..'))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch import nn

from tqdm import tqdm
from datetime import datetime
from PIL import Image

from utils import util_funcs
from models.model_utils import get_dataset, get_model


_NOW = datetime.now().strftime('%Y_%m_%d__%H_%M')


def _save_tensors(exp_name, image_in, image_out, sample_size=5):

    _save_root = os.path.join(args.sample_save_path, exp_name, '_save_tensors')
    os.makedirs(_save_root, exist_ok=True)

    utils.save_image(
        torch.cat([image_in[:sample_size], image_out[:sample_size]], 0),
        os.path.join(_save_root, f'sample_{_NOW}.png'),
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )


def tensor2img(tensor):
    ndarr = tensor.clone().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def _validate_args(checkpoint_path):
    args_file = os.path.dirname(checkpoint_path) + '/args.txt'
    if not os.path.isfile(args_file):
        print(f'cannot locate "args.txt" in {args_file}')
        return

    with open(args_file) as fp:
        kw_strs = [line.split(' : ') for line in fp.readlines()]
        kw_dict = {l[0]: l[1].strip(' \n') for l in kw_strs}

    msg = ''
    FIELDS_TO_SKIP = {'today_str', 'num_workers', 'seed', 'ckpt_epoch', 'experiment_name', ''}
    for k, v in kw_dict.items():
        if k in FIELDS_TO_SKIP:
            continue
        elif not hasattr(args, k):
            msg += f'"{k}" is in the args.txt file but not such parameter exits in input \n'
        elif v != str(getattr(args, k)):
            msg += f'"{k}" has input value of "{getattr(args, k)}" but in checkpoint it has value "{v}"\n'

    if msg != '':
        print(f'{30*"="}\nparameter inconsistency found')
        print(msg[:-1])
        print(f'{30*"="}')


def get_PSNR(size, device, dataset, data_path, num_workers, num_embeddings, architecture, ckpt_epoch, neighborhood, selection_fn, embed_dim, **kwargs):
    print('setting up dataset')
    _, test_dataset = get_dataset(dataset, data_path, size)

    print('creating data loaders')
    test_loader = DataLoader(test_dataset, batch_size=kwargs['vae_batch'], shuffle=True, num_workers=num_workers)

    experiment_name = util_funcs.create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn, size, **kwargs)
    checkpoint_name = util_funcs.create_checkpoint_name(experiment_name, ckpt_epoch)
    checkpoint_path = f"{kwargs['checkpoint_path']}/{checkpoint_name}"
    _validate_args(checkpoint_path)

    print('Calculating PSNR for: {}'.format(checkpoint_name))

    print('Loading model')
    model = get_model(architecture, num_embeddings, device, neighborhood, selection_fn, embed_dim, parallel=False, **kwargs)
    model.load_state_dict(torch.load(os.path.join('..', checkpoint_path), map_location='cuda:0'), strict=False)
    model = model.to(device)
    model.eval()

    mse = nn.MSELoss()
    MAX_i = 255
    to_MAX = lambda t: (t+1) * MAX_i / 2
    psnr_term = 20 * torch.log10(torch.ones(1, 1)*MAX_i)

    # calculate PSNR over test set
    sparsity = 0
    top_percentiles = 0
    num_zeros = 0
    psnrs = 0
    done = 0

    for batch in tqdm(test_loader, desc='Calculating PSNR'):
        with torch.no_grad():
            img = batch[0].to(device)
            out, _, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros = model(img)

        if psnrs == 0 and os.path.isdir(args.sample_save_path):  # save the first test batch
            _save_tensors(experiment_name, img, out, )

        cur_psnr = psnr_term.item() - 10 * torch.log10(mse(to_MAX(out), to_MAX(img)))

        # Gather data
        psnrs += cur_psnr
        sparsity += norm_Z.mean()
        top_percentiles += top_percentile.mean()
        num_zeros += num_zeros.mean()
        done += 1

    # Dump results
    print('sparsity: {}'.format(sparsity))
    print('done: {}'.format(done))
    print('(sparsity/float(done)).item(): {}'.format((sparsity/float(done)).item()))
    avg_psnr = (psnrs/float(done)).item()
    avg_top_percentiles = (top_percentiles/float(done)).item()
    avg_num_zeros = (num_zeros/float(done)).item()
    avg_spasity = (sparsity/float(done)).item()

    print('#'*30)
    print('Experiment name: {}'.format(experiment_name))
    print('Epoch name: {}'.format(ckpt_epoch))
    print('avg_psnr: {}'.format(avg_psnr))
    print('is_quantize_coefs: {}'.format(kwargs['is_quantize_coefs']))

    # dump params and stats into a JSON file
    _save_root = os.path.join(args.sample_save_path, experiment_name)
    os.makedirs(_save_root, exist_ok=True)
    with open(f'{_save_root}/eval_result_{_NOW}.json', 'w') as fp:
        json.dump(dict(
            # --- params
            dataset=args.dataset,
            selection_fn=args.selection_fn,
            embed_dim=args.embed_dim,
            num_atoms=args.num_embeddings,
            image_size=args.size,
            batch_size=args.vae_batch,
            num_strides=args.num_strides,
            num_nonzero=args.num_nonzero,
            normalize_x=args.normalize_x,
            normalize_d=args.normalize_dict,
            epoch=args.ckpt_epoch,
            # --- stats
            psnr=avg_psnr,
            compression=None,  # todo - calculate this
            atom_bits=None,  # todo - calculate this
            non_zero_mean=avg_spasity,
            non_zero_99pct=avg_top_percentiles,
            # tmp=args.tmp,
        ), fp, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    parser.add_argument('--sample_save_path', default='.',
                        type=str, help='a csv file to append the result to')
    args = parser.parse_args()
    util_funcs.seed_generators(args.seed)
    get_PSNR(**vars(args))
