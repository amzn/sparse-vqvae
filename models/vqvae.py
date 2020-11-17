import torch
from torch import nn
from torch.nn import functional as F
from models.quantizers import VanillaQuantize, OMPQuantize, FistaQuantize
from utils.util_funcs import has_value_and_true

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, num_strides=1):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)]

        # Each stride reduces the encoded matrix size by 4
        for i in range(num_strides-1):
            blocks += [
                nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]

        # Block to remove to recreate old decompression results from December
        blocks += [
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        # Use the following block instead of previous block
        # when you want to recreate old decompression results from December
        # blocks += [
        #     # nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
        #     # nn.ReLU(inplace=True),
        #     nn.Conv2d(channel//2, channel, 3, padding=1),
        # ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, num_strides=1
    ):
        super().__init__()

        self.channel = channel
        self.in_channel = in_channel
        self.out_channel = out_channel

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # Block to remove to recreate old decompression results from December
        blocks.extend(
            [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
        )

        for i in range(num_strides-1):
            blocks.extend( [
                nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
            )

        blocks.extend(
            [
                nn.ConvTranspose2d(
                    channel // 2, out_channel, 4, stride=2, padding=1
                ),
            ]
            )

        # Use the following block instead of previous block
        # when you want to recreate old decompression results from December
        # blocks.append(
        #         nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
        #     )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        num_nonzero=1,
        neighborhood=1,
        selection_fn='omp',
        alpha=0.1,
        num_strides=1,
            **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.channel = channel
        self.alpha = alpha

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=2, num_strides=num_strides)
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)

        if selection_fn == 'omp':
            print('Using OMP selection function')
            self.quantize = OMPQuantize(embed_dim, n_embed, num_nonzero=num_nonzero, neighborhood=neighborhood, **kwargs)
        elif selection_fn == 'fista':
            print('Using fista selection function')
            self.quantize = FistaQuantize(embed_dim, n_embed, decay=decay, alpha=self.alpha,  **kwargs)
        elif selection_fn == 'vanilla':
            print('Using vanilla selection function')
            self.quantize = VanillaQuantize(embed_dim, n_embed, decay=decay, **kwargs)
        else:
            raise ValueError('Got an illegal selection function: {}'.format(selection_fn))

        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=2,
            num_strides=num_strides
        )

    def forward(self, input):
        quant, diff, _, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros

    def encode(self, input):
        enc = self.enc(input)

        quant = self.quantize_conv(enc)
        quant, diff, id, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros = self.quantize(quant)

        return quant, diff, id, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros

    def decode(self, quant):
        dec = self.dec(quant)

        return dec

    def decode_code(self, code):
        """
        Given an atom id map - look up the atoms and decode the map
        :param code: Tensor. Matrix of dictionary atom ids.
        For Sparse code the dimensions should be: (Batch, Sparse code, Width, Height)

        :return: Tensor. The result of decoding the given id map
        """
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)

        dec = self.decode(quant)

        return dec

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.quantize = self.quantize.to(*args, **kwargs)
        return self