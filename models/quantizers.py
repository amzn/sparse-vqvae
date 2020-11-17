import torch
from torch import nn
from torch.nn import functional as F
# from torch.nn.functional import normalize

from utils.util_funcs import has_value_and_true
from utils.pyfista import FistaFunction
from utils.pyomp import Batch_OMP

import numpy as np


class FistaQuantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, neighborhood=2, num_workers=4, alpha=0.1, **kwargs):
        """

        :param dim: Int. Size of latent space.
        :param n_embed: Int. Size of dictionary.
        :param alpha: Float. Fista shrinkage value.
        """
        super().__init__()

        self.num_workers=num_workers
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.neighborhood = neighborhood
        self.alpha = alpha

        # Dictionary tensor
        self.dictionary = torch.nn.Parameter(torch.randn(dim, n_embed, requires_grad=True))

        self.normalize_dict = has_value_and_true(kwargs, 'normalize_dict')  # Normalize dictionary flag
        self.normalize_z = has_value_and_true(kwargs, 'normalize_z')  # Normalize sparse code flag
        self.normalize_x = has_value_and_true(kwargs, 'normalize_x')  # Normalize quantization input flag
        self.is_enforce_sparsity = has_value_and_true(kwargs, 'is_enforce_sparsity')  # Flag to enforce sparsity by selecting top K sparse code values
        self.is_quantize_coefs = has_value_and_true(kwargs, 'is_quantize_coefs')  # Flag to quantize sparse code for compression
        self.backward_dict = has_value_and_true(kwargs, 'backward_dict')  # Flag to backprop with respect to the dictionary
        self.sample_gradients = kwargs['sample_gradients']  # Number of gradients to use when backproping through the dictionary
        self.use_backwards_simd = kwargs['use_backwards_simd']  # Use matrix backprop instead of loop backprop

        self.normalization_flag = self.normalize_dict
        self.debug = False

    def forward(self, input):
        """
        Denote dimensions:
        B - number of patches
        d - size of latent space
        D - size of dictionary

        :param input: Signal to get the sparse code for
        :return: quantized input
        """

        permuted_input = input.permute(0, 2, 3, 1)
        flatten = permuted_input.reshape(-1, self.dim)  # Shape: Bxd

        if self.normalize_x:
            flatten = F.normalize(flatten, p=2, dim=1)

        if self.normalization_flag and self.normalize_dict:
            with torch.no_grad():  # Cannot directly change a module Parameter outside of no_grad
                self.dictionary.data = self.dictionary.data.__div__(torch.norm(self.dictionary.data,p=2,dim=0))  # Shape: dXD

            if not self.training:
                self.normalization_flag = False

        sparse_code, num_fista_steps = FistaFunction.apply(flatten.t(), self.dictionary, self.alpha, 0.01, -1, False,
                                                           self.sample_gradients, self.use_backwards_simd)
        if self.debug:
            # We print this to understand what the range of the sparse code is to know when we quantize it in test time
            print('Sparse code value range: min: {}  |  max: {}'.format(sparse_code.min(), sparse_code.max()))

        if self.training and self.normalize_z and sparse_code.abs().max() > np.finfo(float).eps * 10: #TODO: Decide if we continue skipping or find another solution like reduce alpha and run again
            with torch.no_grad():  # We can normalize the coefs with learning through them but we choose not to for symmetry with embeddings
                sparse_code.data = sparse_code.data.__div__(torch.norm(sparse_code.data,p=2,dim=0))  # Shape: BXD

        if self.debug:
            print('Sparse code average L0 norm: {}'.format(sparse_code.norm(0, 0).mean()))

        # This is only used when calculating PSNR to control over the compression rate
        if not self.training and self.is_enforce_sparsity:
            sparse_code = self.enforce_sparsity(sparse_code)

        # Quantize sparse code to only use a set number of bits
        if self.is_quantize_coefs:
            # print('Quantizing sparse code coefficients')
            sparse_code = self.hardcode_quantize(sparse_code)

        # Apply sparse code to input to get quantization of it
        quantize = sparse_code.t().float().mm(self.dictionary.t()).to(flatten.device)
        quantize = quantize.view(*permuted_input.shape)

        # We reshape the sparse code as well to conform to patches
        reshapes = list(permuted_input.shape)
        reshapes[-1] = sparse_code.size()[0]
        ids = sparse_code.t().view(*reshapes)

        if self.backward_dict:
            quantization_diff_for_encoder = (quantize.detach() - permuted_input).pow(2).mean()
            quantization_diff_for_dictionary = (quantize - permuted_input.detach()).pow(2).mean()
            quantize = permuted_input + (quantize - permuted_input).detach()
        else:
            # If we don't want to backprop through the dictionary we simply duplicate the quantization_diff_for_encoder
            # and detach to prevent double backprop
            quantization_diff_for_encoder = (quantize.detach() - permuted_input).pow(2).mean()
            quantization_diff_for_dictionary = (quantize.detach() - permuted_input).pow(2).mean().detach()
            quantize = permuted_input + (quantize - permuted_input).detach()

        # Reporting like this and not in a single object because PyTorch throws a fit
        norm_0 = sparse_code.norm(0, 0)
        num_quantization_steps = num_fista_steps.detach()
        mean_D = self.dictionary.abs().mean().detach()
        mean_Z = sparse_code.abs().mean().detach()
        norm_Z = norm_0.mean().detach()
        topk_num = max(1, int(len(norm_0)*0.01))

        top_percentile = norm_0.topk(topk_num).values.min().detach()
        num_zeros = (norm_0==0).int().sum().float().detach()
        if self.debug:
            print('num zero: {}'.format(num_zeros))

        return quantize.permute(0, 3, 1, 2), [quantization_diff_for_encoder, quantization_diff_for_dictionary], ids.permute(0, 3, 1, 2), num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros

    def enforce_sparsity(self, sparse_code, sparsity_size=10):
        """
        This function is used to enforce a certain sparsity on the input sparse code by keeping only the top
        sparsity_size values non-zero.
        :param sparse_code: Tensor. Sparse code that we want to enforce sparsity on
        :param sparsity_size: Int. Hard limit on the non-zeros we allow in the sparse code
        :return: Tensor. The sparse_code with only the top sparsity_size values not zeroed out.
        """
        tmp_coefs1 = torch.zeros(sparse_code.size()).to(sparse_code.device)
        tmp_coefs2 = torch.zeros(sparse_code.size()).to(sparse_code.device)
        tops = torch.topk(sparse_code.abs(), sparsity_size, 0)
        torch.gather(sparse_code.detach(), 0, tops[1], out=tmp_coefs1)
        tmp_coefs2.scatter_(0, tops[1], tmp_coefs1)
        sparse_code = tmp_coefs2
        return sparse_code

    def hardcode_quantize(self, sparse_code, min_val=-0.55, max_val=0.55, bits=8):
        """
        Clamps the sparse code in the range (min_val, max_val) and quantizes to limited number of bits.
        :param sparse_code: Tensor. Sparse code to be quantized.
        :param min_val: Float. Lower range boundary.
        :param max_val: Float. Upper range boundary.
        :param bits: Int. Number of bits to qunatize to.
        :return: Tensor. Quantized sparse code.
        """
        sparse_code=sparse_code.clamp(min_val, max_val)
        sparse_code -= min_val
        sparse_code /= (max_val - min_val)
        sparse_code *= 2**bits
        sparse_code = sparse_code.round()
        sparse_code /= 2**bits
        sparse_code *= (max_val-min_val)
        sparse_code += min_val
        return sparse_code

    def embed_code(self, sparse_code):
        """
        Performs a de-quantization operation for the given sparse code
        :param sparse_code: Tensor. Sparse code we desire to use. Dimensions: (Batch, Sparse code, Width, Height)
        :return: Tensor. Linear combination of the dictionary based on given sparse code
        """

        # Transform to (Batch, Width, Height, Sparse code), then flatten the first three dimensions
        permuted_sparse_code = sparse_code.permute(0, 2, 3, 1)
        aligned_sparse_code = permuted_sparse_code.contiguous().view(-1, sparse_code.size()[1])

        # Project dictionary on sparse code to create latent image
        result = aligned_sparse_code.mm(self.dictionary.t()).t()

        # Reshape latent image to (Batch, Width, Height, Latent)
        reshaped_result = result.view(-1, sparse_code.size()[0], sparse_code.size()[2], sparse_code.size()[3])
        permuted_results = reshaped_result.permute(1, 2, 3, 0)

        return permuted_results


class OMPQuantize(nn.Module):
    def __init__(self, dim, n_embed, num_nonzero=1, eps=1e-9, num_workers=4, **kwargs):
        super().__init__()

        self.num_workers = num_workers
        self.dim = dim
        self.n_embed = n_embed
        self.num_nonzero = num_nonzero
        self.eps = eps

        # Dictionary tensor
        self.dictionary = torch.nn.Parameter(torch.randn(dim, n_embed, requires_grad=True))
        self.normalize_dict = has_value_and_true(kwargs, 'normalize_dict')

        self.normalize_dict = has_value_and_true(kwargs, 'normalize_dict')  # Normalize dictionary flag
        self.is_quantize_coefs = has_value_and_true(kwargs, 'is_quantize_coefs')  # Flag to quantize sparse code for compression
        self.normalize_x = has_value_and_true(kwargs, 'normalize_x')  # Normalize quantization input flag
        self.backward_dict = has_value_and_true(kwargs, 'backward_dict')  # Flag to backprop with respect to the dictionary

        self._quantize_bits = 8
        self._quantize_max_val = 0.55

    def forward(self, _input):

        permuted_input = _input.permute(0, 2, 3, 1)
        flatten = permuted_input.reshape(-1, self.dim)  # Shape: Bxd

        if self.normalize_x:
            flatten = F.normalize(flatten, p=2, dim=1)

        if self.normalize_dict:
            with torch.no_grad():  # Cannot directly change a module Parameter outside of no_grad
                self.dictionary.data = self.dictionary.data.__div__(torch.norm(self.dictionary.data,p=2,dim=0))  # Shape: dXD

        with torch.no_grad():  # OMP selection process
            sparse_code = Batch_OMP(flatten.t(), self.dictionary, self.num_nonzero, tolerance=self.eps)

        # Quantize sparse code to only use a predefined number of bits
        if self.is_quantize_coefs:
            # print('Quantizing sparse code coefficients')
            sparse_code = self.hardcode_quantize(sparse_code)

        # Apply sparse code to input to get quantization of it
        quantize = sparse_code.t().float().mm(self.dictionary.t()).to(_input.device)
        quantize = quantize.view(*permuted_input.shape)

        # We reshape the sparse code as well to conform to patches
        reshapes = list(permuted_input.shape)
        reshapes[-1] = sparse_code.size()[0]
        ids = sparse_code.t().view(*reshapes)

        if self.backward_dict:
            quantization_diff_for_encoder = (quantize.detach() - permuted_input).pow(2).mean()
            quantization_diff_for_dictionary = (quantize - permuted_input.detach()).pow(2).mean()
            quantize = permuted_input + (quantize - permuted_input).detach()
        else:
            # If we don't want to backprop through the dictionary we simply duplicate the quantization_diff_for_encoder
            # and detach to prevent double backprop
            quantization_diff_for_encoder = (quantize.detach() - permuted_input).pow(2).mean()
            quantization_diff_for_dictionary = (quantize.detach() - permuted_input).pow(2).mean().detach()
            quantize = permuted_input + (quantize - permuted_input).detach()

        # Reporting like this and not in a single object because PyTorch throws a fit
        norm_0 = sparse_code.norm(0, 0)
        num_quantization_steps = torch.tensor(self.num_nonzero, dtype=torch.float).to(self.dictionary.device).detach()
        mean_D = self.dictionary.abs().mean().detach()
        mean_Z = sparse_code.abs().mean().detach()
        norm_Z = norm_0.mean().detach()
        topk_num = max(1, int(len(norm_0)*0.01))
        top_percentile = norm_0.topk(topk_num).values.min().detach()
        num_zeros = (norm_0 == 0).int().sum().float().detach()

        return quantize.permute(0, 3, 1, 2), [quantization_diff_for_encoder, quantization_diff_for_dictionary], ids.permute(0, 3, 1, 2), num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros

    def embed_code(self, sparse_code):
        """
        Performs a de-quantization operation for the given sparse code
        :param sparse_code: Tensor. Sparse code we desire to use. Dimensions: (Batch, Sparse code, Width, Height)
        :return: Tensor. Linear combination of the dictionary based on given sparse code
        """

        # Transform to (Batch, Width, Height, Sparse code), then flatten the first three dimensions
        permuted_sparse_code = sparse_code.permute(0, 2, 3, 1)
        aligned_sparse_code = permuted_sparse_code.contiguous().view(-1, sparse_code.size()[1])

        # Project dictionary on sparse code to create latent image
        result = aligned_sparse_code.mm(self.dictionary.t()).t()

        # Reshape latent image to (Batch, Width, Height, Latent)
        reshaped_result = result.view(-1, sparse_code.size()[0], sparse_code.size()[2], sparse_code.size()[3])
        permuted_results = reshaped_result.permute(1, 2, 3, 0)

        return permuted_results

    def hardcode_quantize(self, sparse_code):
        """
        Clamps the sparse code in the range (min_val, max_val) and quantizes to limited number of bits.
        :param sparse_code: Tensor. Sparse code to be quantized.
        :return: Tensor. Quantized sparse code.
        """
        min_val = -self._quantize_max_val
        max_val = self._quantize_max_val
        sparse_code = sparse_code.clamp(min_val, max_val)
        sparse_code -= min_val
        sparse_code /= (max_val - min_val)
        sparse_code *= 2 ** self._quantize_bits
        sparse_code = sparse_code.round()
        sparse_code /= 2 ** self._quantize_bits
        sparse_code *= (max_val-min_val)
        sparse_code += min_val
        return sparse_code


class VanillaQuantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, **kwargs):
        """
        :param dim: Int. Size of latent space.
        :param n_embed: Int. Size of dictionary.
        """
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        dictionary = torch.randn(dim, n_embed)
        self.register_buffer('dictionary', dictionary)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('dictionary_avg', dictionary.clone())
        self.normalize_dict = has_value_and_true(kwargs, 'normalize_dict')
        self.normalize_x = has_value_and_true(kwargs, 'normalize_x')

        self.normalization_flag = self.normalize_dict

    def forward(self, _input):
        """
        Denote dimensions:
        B - number of patches
        d - size of latent space
        D - size of dictionary

        :param input: Signal to get the sparse code for
        :return: quantized input
        """

        if self.normalization_flag and self.normalize_dict:  # We don't need the no_grad operation as dictionary if a buffer
            self.dictionary = self.dictionary.div(torch.norm(self.dictionary, p=2, dim=0).expand_as(self.dictionary))

            if not self.training:
                self.normalization_flag = False

        permuted_input = _input.permute(0, 2, 3, 1)
        flatten = permuted_input.reshape(-1, self.dim)  # Shape: Bxd

        if self.normalize_x:
            flatten = F.normalize(flatten, p=2, dim=1)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.dictionary
            + self.dictionary.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*permuted_input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        # This is the EMA dictionary-optimization step
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.dictionary_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.dictionary_avg / cluster_size.unsqueeze(0)
            self.dictionary.data.copy_(embed_normalized)

        diff = (quantize.detach() - permuted_input).pow(2).mean()
        quantize = permuted_input + (quantize - permuted_input).detach()

        # Reporting like this and not in a single object because PyTorch throws a fit
        num_quantization_steps = torch.ones(1).to(self.dictionary.device).detach()
        mean_D = self.dictionary.abs().mean().detach()
        mean_Z = torch.ones(1).to(self.dictionary.device).detach()
        norm_Z = torch.ones(1).to(self.dictionary.device).detach()
        top_percentile = torch.ones(1).to(self.dictionary.device).detach()
        num_zeros = torch.zeros(1).to(self.dictionary.device).detach()

        return quantize.permute(0, 3, 1, 2), [diff, torch.zeros(0).to(self.dictionary.device)], embed_ind, num_quantization_steps, mean_D, mean_Z, norm_Z, top_percentile, num_zeros

    def embed_code(self, embed_id):
        """
        Performs a lookup operation for the given id's atom
        :param embed_id: Int. Dictionary atom to look for
        :return: Tensor. Atom looked for
        """
        return F.embedding(embed_id, self.dictionary.transpose(0, 1))