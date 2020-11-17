import os
import random
import numpy as np
import torch
from argparse import ArgumentParser


osp = os.path
_e_root = '.'
n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
n_cpu = torch.multiprocessing.cpu_count()
NUM_WORKERS = min(n_cpu, max(8, 12 * n_gpu))

data_root = './data'
if osp.isdir('/home/ubuntu/data/imagenet') and osp.isdir('/home/ubuntu/data/cifar10'):
    data_root = '/home/ubuntu/data'  # if data was copied locally


def base_parser(parser: ArgumentParser):
    parser.add_argument('--size', type=int, default=64, help='Resize samples to this size')
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of data loading workers')
    parser.add_argument('--device', default='cuda', type=str, help='Device to run on [cpu | cuda | cuda:device_number]')
    parser.add_argument('--data_path', default=data_root, type=str, help='Path to dataset')
    parser.add_argument('--checkpoint_path', default=osp.join(_e_root, 'checkpoint'),
                        type=str, help='Path where checkpoints are saved')
    parser.add_argument('--summary_path', default=osp.join(_e_root, 'summary'),
                        type=str, help='Path where saved (tensorboard) summries are written')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Name of dataset to use [cifar10 | cifar100 | imagenet]')
    parser.add_argument('-n', '--experiment_name', default='', type=str, help='Name of experiment')

    return parser


def vqvae_parser(parser: ArgumentParser):
    # vqvae training params
    parser.add_argument('--alpha', type=float, default=0.3, help='Fista shrinkage parameter')
    parser.add_argument('-k', '--num_nonzero', type=int, default=2, help='OMP maximal number of nonzero')

    parser.add_argument('--num_epochs', type=int, default=200, help='Train VQ-VAE model for this number of epochs')
    parser.add_argument('--embed_dim', type=int, default=64, help='Size of the latent space')
    parser.add_argument('--sample_gradients', type=int, default=1000, help='Number of patches to backprop against in FISTA. 0 for all')

    parser.add_argument('--vae_batch', type=int, default=128, help='Batch size for the VQ-VAE model')
    parser.add_argument('--lr', type=float, default=3e-4, help='VQ-VAE learning rate')
    parser.add_argument('--dictionary_loss_weight', type=float, default=0.0, help='Weight of the dictionary loss term.')
    parser.add_argument('--sched', type=str, help='Scheduler to use. [cycle | ]. Default empty')
    parser.add_argument('--architecture', default='vqvae', type=str, help='Name of architecture to use [vqvae]')
    parser.add_argument('--num_embeddings', default=512, type=int, help='Number of embeddings in code book')
    parser.add_argument('-sel', '--selection_fn', default='vanilla', type=str, help='Function to select dictionary vectors [omp | vanilla | fista ]')
    parser.add_argument('--neighborhood', default=1, type=int, help='Only relevant for the OMP.')
    parser.add_argument('--checkpoint_freq', default=5, type=int, help='Checkpoint model every this number of epochs.')
    parser.add_argument('--backward_dict', default=1, type=int, help='1 to do backprop w.r.t. the dictionary, 0 otherwise.')
    parser.add_argument('-stride', '--num_strides', default=2, type=int, help='Number of stride blocks, every block reduces the size of the quantized image by half.')
    parser.add_argument('--use_backwards_simd', default=True, type=bool, help='Flag to use matrix version of the FISTA backprop.')
    parser.add_argument('--download', default=False, type=bool, help='Flag to download the dataset if needed.')

    parser.add_argument('--parallelize', dest='parallelize', action='store_true', help='Flag to use DataParallel')
    parser.add_argument('--no_parallelize', dest='parallelize', action='store_false', help='Flag not to use DataParallel')
    parser.set_defaults(parallelize=True)

    parser.add_argument('--normalize_dict', dest='normalize_dict', action='store_true', help='Flag to normalize dictionary')
    parser.add_argument('--no_normalize_dict', dest='normalize_dict', action='store_false', help='Flag not to normalize dictionary')
    parser.set_defaults(normalize_dict=True)

    parser.add_argument('--normalize_z', dest='normalize_z', action='store_true', help='Flag to normalize found sparse code')
    parser.add_argument('--no_normalize_z', dest='normalize_z', action='store_false', help='Flag not to normalize found sparse code')
    parser.set_defaults(normalize_z=False)

    parser.add_argument('--normalize_x', dest='normalize_x', action='store_true', help='Flag to normalize quantization input')
    parser.add_argument('--no_normalize_x', dest='normalize_x', action='store_false', help='Flag not to normalize quantization input')
    parser.set_defaults(normalize_x=True)

    # Training evaluation and sampling parameters
    parser.add_argument('--eval_iter', default=1, type=int, help='Eval every [value] iterations')
    parser.add_argument('--sampling_iter', default=25, type=int, help='Sample every [value] batches')
    parser.add_argument('--sample_size', default=25, type=int, help='Number of images to sample every [value] batches')

    # Test parameters
    parser.add_argument('--is_enforce_sparsity', dest='is_enforce_sparsity', action='store_true',
                        help='Flag to select only top-K sparse code values, needed only for FISTA')
    parser.set_defaults(is_enforce_sparsity=False)

    parser.add_argument('--is_quantize_coefs', dest='is_quantize_coefs', action='store_true',
                        help='Flag to quantize sparse code coefficients for compression')
    parser.set_defaults(is_quantize_coefs=False)

    return parser


def code_extraction_parser(parser: ArgumentParser):
    parser.add_argument('--ckpt_epoch', type=int, default=200, help='Epoch number of the VQVAE model to load')

    return parser


def pixelsnail_parser(parser: ArgumentParser):
    parser.add_argument('--pixelsnail_batch', type=int, default=256, help='Size of PixelSnail batch')
    parser.add_argument('--pixelsnail_epoch', type=int, default=420, help='Train PixelSnail model for this number of epochs')
    parser.add_argument('--hier', type=str, default='bottom', help='Used for cascaded VQ-VAE. ', choices='bottom')  # FIXME - Use only `bottom` for now
    parser.add_argument('--pixelsnail_lr', type=float, default=3e-4, help='PixelSnail learning rate')
    parser.add_argument('--pixelsnail_channel', type=int, default=256, help='Number of channels to expand to in the PixelSnail architecture')
    parser.add_argument('--pixelsnail_n_res_block', type=int, default=4, help='Number of residual blocks in PixelSnail')
    parser.add_argument('--pixelsnail_n_res_channel', type=int, default=256, help='Number of channels to expand to in residual blocks in PixelSnail')
    parser.add_argument('--pixelsnail_n_out_res_block', type=int, default=0, )
    parser.add_argument('--pixelsnail_n_cond_res_block', type=int, default=3)
    parser.add_argument('--pixelsnail_dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--pixelsnail_sched', type=str)
    parser.add_argument('--pixelsnail_ckpt', type=str, help='PixelSnail checkpoint to continue training from')

    return parser


def sampling_parser(parser: ArgumentParser):
    parser.add_argument('--pixelsnail_ckpt_epoch', type=int, default=420, help='PixelSnail epoch to load')
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--num_threads', type=int, default=3, help='Number of threads to multithread the sampling on')

    return parser


def create_experiment_name(architecture, dataset, num_embeddings, neighborhood, selection_fn, size, lr=None, **kwargs):
    additional_remarks = ''

    if kwargs['experiment_name'] == '':
    # if 'old_experiment_name_format' in kwargs and kwargs['old_experiment_name_format']:
        if hasattr(kwargs, 'experiment_name') and len(kwargs['experiment_name']) > 0:
            additional_remarks += '_experiment_name_{}'.format(kwargs['experiment_name'])

        if has_value_and_true(kwargs, 'normalize_dict'):
            pass  # additional_remarks += '_normalize_dict_{}'.format(True)
        else:
            additional_remarks += '_normalize_dict_{}'.format(False)

        if has_value_and_true(kwargs, 'normalize_x'):
            pass  # additional_remarks += '_normalize_x_{}'.format(True)
        else:
            additional_remarks += '_normalize_x_{}'.format(False)

        # if has_value_and_true(kwargs, 'normalize_z'):
        #     additional_remarks += '_normalize_z_{}'.format(True)
        # else:
        #     additional_remarks += '_normalize_z_{}'.format(False)

        additional_remarks += '_size_{}'.format(size)

        if hasattr(kwargs, 'ckpt_epoch'):
            additional_remarks += '_ckpt_epoch_{}'.format(kwargs['ckpt_epoch'])

        if hasattr(kwargs, 'sample_gradients'):
            additional_remarks += '_sample_gradients_{}'.format(kwargs['sample_gradients'])

        if hasattr(kwargs, 'backward_dict') and selection_fn in ('fista', 'omp'):
            additional_remarks += '_backward_dict_{}'.format(kwargs['backward_dict'])

        if hasattr(kwargs, 'lr'):
            additional_remarks += 'lr_{}'.format(kwargs['lr'])
        # else:
        #     additional_remarks += 'lr_{}'.format(lr)

        alpha = kwargs['alpha']
        experiment_name = f'{architecture}_{dataset}_num_embeddings{num_embeddings}_neighborhood{neighborhood}_selectionFN{selection_fn}_alpha{alpha}' + additional_remarks
    else:
        experiment_name = kwargs['experiment_name']

    return experiment_name


def create_checkpoint_name(experiment_name, epoch_ind):
    return f'{experiment_name}/{str(epoch_ind).zfill(3)}.pt'


def has_value_and_true(dictionary, key):
    return key in dictionary and dictionary[key]


def seed_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


