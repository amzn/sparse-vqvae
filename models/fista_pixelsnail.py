# Copyright (c) Xi Chen
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Borrowed from https://github.com/neocxi/pixelsnail-public and ported it to PyTorch

# from math import sqrt
# from functools import partial, lru_cache

# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
from models.pixelsnail import *
# import joblib


class FistaPixelSNAIL(PixelSNAIL):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0,
    ):
        super().__init__(shape,
        n_class,
        channel,
        kernel_size,
        n_block,
        n_res_block,
        res_channel,
        attention=attention,
        dropout=dropout,
        n_cond_res_block=n_cond_res_block,
        cond_res_channel=cond_res_channel,
        cond_res_kernel=cond_res_kernel,
        n_out_res_block=n_out_res_block)

        self.eps = np.finfo(float).eps * 10

        # Override base PixelSnail out module
        out = []

        for i in range(n_out_res_block):
            out.append(GatedResBlock(channel, res_channel, 1))

        self.out = nn.Sequential(*out)

        # Declare network heads
        self.sampling_head = nn.Sequential(*[nn.ELU(inplace=False), WNConv2d(channel, n_class, 1)])
        self.nonzeros_head = nn.Sequential(*[nn.ELU(inplace=False), WNConv2d(channel, n_class, 1)])

        # Create a joined tensor for the mu and sigma for the reparametrization trick
        self.reparamaterization_head = nn.Sequential(*[nn.ELU(inplace=False), WNConv2d(channel, n_class*n_class+n_class, 1)])

    @staticmethod
    def reparameterize(mu, sigma):
        """
        Reparameterization trick.

        Dimension notation:
        B: Batch
        S: Sparse code
        W: Width
        H: Height

        :param mu: Float Tensor. (B, W, H, S). Predicted median for coefficient per atom per patch
        :param sigma: Float Tensor. (B, W, H, S, S). Predicted covariance for coefficient per atom per patch
        :return: Float Tensor. (B, W, H, S). Sampled coefficients per atom per patch.
        """

        # We change the view of the parameters otherwise having unique dimensions for the Width and Height
        # would interfere with the product operation
        mu_view = mu.contiguous().view([-1, mu.size()[-1]])
        sigma_view = sigma.contiguous().view([-1, sigma.size()[-2], sigma.size()[-1]])
        theta_0 = torch.randn(mu_view.size()).to(mu_view.device)

        theta_1 = (theta_0 - mu_view).unsqueeze(1).bmm(sigma_view)
        theta_1 = theta_1.view(mu.size())
        return theta_1

    def prepare_inputs(self, Z):
        """
        Extracts ground-truth from given sparse code. Useful only in training

        Dimension notation:
        B: Batch
        S: Sparse code
        W: Width
        H: Height

        :param Z: Float Tensor. (B, S, W, H). Given sparse code for every patch
        :return:
         1. used_atoms_mask, Bool Tensor. (B, S, W, H). Map of atoms which are non-zero
         2. gt_num_nonzeros, Long Tensor. (B, W, H). Number of non-zero atoms for each patch
        """
        used_atoms_mask = torch.abs(Z) > self.eps
        gt_num_nonzeros = used_atoms_mask.sum(1)  # sum over the sparse code dimension

        return used_atoms_mask, gt_num_nonzeros

    def forward(self, Z, used_atoms_mask, gt_num_nonzeros):
        """
        Dimension notation:
        B: Batch
        S: Sparse code
        W: Width
        H: Height

        :param Z: Float Tensor. (B, S, W, H). Map of sparse code patches
        :param used_atoms_mask: Bool Tensor. (B, S, W, H). Map of atoms which are non-zero
        :param gt_num_nonzeros: Long Tensor. (B, W, H). Number of non-zero atoms for each patch
        """

        batch, n_class, height, width = Z.size()
        assert n_class == self.n_class

        horizontal = shift_down(self.horizontal(Z))
        vertical = shift_right(self.vertical(Z))
        out = horizontal + vertical

        background = self.background[:, :, :height, :].expand(batch, 2, height, width)

        for block in self.blocks:
            out = block(out, background)

        out = self.out(out)
        sampled_atoms = self.sampling_head(out)  # Output a probability for each atom to be used or not
        sampled_num_nonzeros = self.nonzeros_head(out)  # Classify how many non-zero atoms there are

        # Binary mask of all selected atoms, per patch
        expanded_matrix_nonzero_inds_mask = used_atoms_mask.unsqueeze(dim=1).repeat(1, self.n_class, 1, 1, 1)

        # Output the mu and sigma from which we sample the coefficients of non-zero atoms
        reparametarization_results = self.reparamaterization_head(out)
        mu = reparametarization_results[:, :self.n_class, :, :]
        sigma_vector = reparametarization_results[:, self.n_class:, :, :]
        sigma_matrix = sigma_vector.view([sigma_vector.size()[0], self.n_class, self.n_class, sigma_vector.size()[2], sigma_vector.size()[3]])

        # (B, S, W, H) -> (B, W, H, S): Push the coefficient data dimension to last for the reparameterization trick
        permuted_mu = mu.permute(0, 2, 3, 1)
        # (B, S, S, W, H) -> (B, W, H, S, S): Push the coefficient data dimension to last for the reparameterization trick
        permuted_sigma_matrix = sigma_matrix.permute(0, 3, 4, 1, 2)

        permuted_coefficients = self.reparameterize(permuted_mu, permuted_sigma_matrix)

        # Un-permute the results: (B, W, H, S) -> (B, S, W, H)
        coefficients = permuted_coefficients.permute(0, 3, 1, 2)

        # Apply used_atoms_mask
        masked_coefficients = coefficients * used_atoms_mask

        return sampled_atoms, sampled_num_nonzeros, masked_coefficients
