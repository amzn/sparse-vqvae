"""
Calculates PSNR for JPG with respect to a dataset.
Note: Currently hard-coded for Cifar10
"""
import sys
import os
sys.path.append(os.path.abspath('..'))
import argparse
import numpy as np
import torch
import cv2
from collections import namedtuple
from PIL import Image
from tqdm import tqdm

from utils import util_funcs
from models.model_utils import get_dataset


result_tuple = namedtuple('JPEG_res', ['quality', 'ratio', 'psnr', ])


def tensor2img(tensor):
    ndarr = tensor.clone().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def get_PSNR(quality, dataset, size):
    total_psnr = 0
    total_steps = 0
    MAX_i = 255
    psnr_term = 20 * np.log10(np.ones(1) * MAX_i)
    compression_ratio = 0

    _tqdm = tqdm(dataset, desc=f'Quality: {quality}')
    for batch in _tqdm:
        # Load data
        image = (batch[0] + 1) * MAX_i / 2  # from [-1,1] to [0, 255]
        assert image.max() <= MAX_i and image.min() >= 0, f'bad image valuse, in [{image.max()}, {image.min()}]'
        if image.shape[0] == 3:
            image = image.transpose(0, 2)
        rgb_image = np.array(image, dtype=np.uint8)
        open_cv_image = rgb_image[:, :, ::-1].copy()

        # Translate to and back from jpg
        jpg_str = cv2.imencode('.jpg', open_cv_image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tostring()
        np_arr = np.fromstring(jpg_str, np.uint8)
        decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Calculate encoded length
        raw_size = size * size * 3
        # bmp_size = len(cv2.imencode('.bmp', open_cv_image)[1])
        jpg_size = len(jpg_str)

        # Calculate compression ratio and PSNR
        compression_ratio += raw_size / float(jpg_size)

        mse = np.mean(np.square(np.array(decoded_img) - rgb_image))
        psnr = psnr_term - 10 * np.log10(mse)
        total_psnr += psnr
        total_steps += 1.0

        if total_steps % 1000 == 0:
            _tqdm.set_postfix({'psnr': np.round(total_psnr/total_steps, 2), 'ratio': compression_ratio/total_steps })

    print('Calculating for JPG with quality measure: {}'.format(quality))
    print('PSNR: {}'.format(total_psnr / total_steps))
    print('compression_ratio: {}'.format(compression_ratio / total_steps))

    return result_tuple(
        quality=quality,
        psnr=total_psnr / total_steps,
        ratio=compression_ratio / total_steps,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = util_funcs.base_parser(parser)
    parser = util_funcs.vqvae_parser(parser)
    parser = util_funcs.code_extraction_parser(parser)
    args = parser.parse_args()

    print('setting up datasets')
    _, test_dataset = get_dataset(args.dataset, args.data_path, args.size)

    all_res = list()
    for quality in range(0, 110, 10):
        res = get_PSNR(quality, test_dataset, args.size)
        all_res.append(res)

    [print(f'{r.quality}, {r.psnr}, {r.ratio}') for r in all_res]
    with open('/tmp/jpeg_res.csv', 'w') as fp:
        [fp.write(f'{r.quality}, {r.psnr}, {r.ratio} \n') for r in all_res]
