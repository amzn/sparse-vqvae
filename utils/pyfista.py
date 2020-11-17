import torch
from sklearn.decomposition import DictionaryLearning
from torch.autograd import Function
import numpy as np

from time import time

def get_largest_eigenvalue(X):
    eigs = torch.eig(X, eigenvectors=False).eigenvalues
    max_eign = eigs.max(dim=0)
    return max_eign.values[0]


def shrink_function(Z, cutoff):
    '''
    Shrink function is (max(0, abs(Z) - cutoff) * sign(Z)
    :return: Shrinked Z
    '''
    cutted = apply_cutoff(Z, cutoff)
    maxed = apply_relu(Z, cutted)
    signed = apply_sign(Z, maxed)
    return signed


def apply_sign(Z, maxed):
    signed = maxed * torch.sign(Z)
    return signed


def apply_relu(Z, cutted):
    maxed = torch.max(cutted, torch.zeros(Z.size(), dtype=Z.dtype).to(Z.device))
    return maxed

def apply_cutoff(Z, cutoff):
    cutted = torch.abs(Z) - cutoff
    return cutted


def reconstruction_distance(D, cur_Z, last_Z):
    distance = torch.norm(D.mm(last_Z - cur_Z), p=2, dim=0) / torch.norm(D.mm(last_Z), p=2, dim=0)
    max_distance = distance.max()
    return distance, max_distance


class FistaFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, Wd, alpha=0.1, tolerance=0.01, max_steps=-1, debug=False, sample_gradients=0,
                use_backwards_simd=True):
        """
        Dimension notation:
            B - Number of samples. Usually number of patches in image times batch size
            K - Number of atoms in dictionary
            d - Dimensionality of atoms in dictionary


        :param input: Input - Signal to find sparse coding against. Dimenions: d x B
        :param Wd: Dictionary - Tensor of atoms we want to get a sparse linear combination of. Dimensions: d x K
        :param alpha:  Float. Sparsity weight
        :param tolerance: Float. Stop optimization once the improvement drops below this term.
        :param max_steps: Int. Stops optimizaiton after max_steps iterations. Ignored if < 0.
        :param debug: Bool. Output debug message if True.
        :param sample_gradients: Int. Limit number of random patches to perform backward propagation against. 0 to select all.
        :param use_backwards_simd: Bool. True if a faster SIMD based implementation should be used for backwards.

        :return: Z: linear coefficients for Sparse Code solution. Dimensions: K x B

        """
        start = time()
        with torch.no_grad():
            Z, num_steps = FISTA(input, Wd, alpha, tolerance, max_steps, debug)
        ctx.save_for_backward(input, Wd, Z)
        ctx.sample_gradients = sample_gradients
        ctx.use_backwards_simd = use_backwards_simd

        if debug:
            print('Done forward in {} seconds'.format(time()-start))
        return Z, torch.ones(1, 1).to(Z.device).detach() * num_steps

    @staticmethod
    def backward_simd(Z, Wd, grad_output, patch_index_list=None):
        """
        Uses an efficient Single-Instruction-Multiple-Data implementation,
        to deriving gradients of the dictionary with respect to the selected sparse code.

        Variable explanation:
        Z: Selected sparse code (results of FISTA). Dimensions: K x B
        Wd: Dictionary - Tensor of atoms we want to get a sparse linear combination of. Dimensions: d X K
        grad_output: Gradients recieved for the dictionary in respect to down-stream task (reconstruction, etc.)
        patch_index_list: List[int] - list of sampled patched indices (optional)
        """
        # B is the batch size; when sampling is used, this is reduced to the *sampled* batch size
        # K is the total number of atoms within the dictionary
        # D is the dimensionality of each atom within the dictionary
        B = Z.shape[1]
        D, K = Wd.shape

        # Epsilon values, used within the algorithm for thresholding zero float values and petrubrations for
        # numerical stability
        eps = np.finfo(float).eps * 10
        peturb_eps = eps * 10000000000

        # Make sure inputs refer to sampled patched only
        if patch_index_list is not None:
            B = len(patch_index_list)
            matrix_Z = Z[:, patch_index_list]
            matrix_grad_output = grad_output[:, patch_index_list]

        # matrix_beta ~ (B, K) ; Here B is always the original batch size, even when sampling is enabled
        # Holds the results of backwards logic, used to weight the gradients
        matrix_beta = torch.zeros(grad_output.size()).type(grad_output.type()).t().to(grad_output.device)

        # matrix_nonzero_inds_mask ~ (B, K)
        # Binary mask of all selected atoms, per patch
        matrix_nonzero_inds_mask = torch.abs(matrix_Z.t()) > eps

        # matrix_num_nonzero ~ (B)
        # How many codes each patch have selected. For each patch we keep [0,K].
        matrix_num_nonzero = torch.sum(matrix_nonzero_inds_mask, dim=1)

        # grad_mask ~ (B)
        # A binary mask of 1 if the patch have selected any sparse codes at all and 0 otherwise
        matrix_grad_mask = matrix_num_nonzero > 0

        # amount_of_padding_dimensions ~ scalar
        # K_pad ~ scalar
        # D_pad ~ scalar
        # K_pad: Each patch selects a different amount of codes, so we must pad the following calculations
        # with extra "fake codes" the algorithm can choose.
        # Here we measure a new K_pad (padded K) with these extra fake dimensions, and determine it to be
        # the orignial K (#of atoms) + the difference between patch that selected the least codes and the total
        # number of possibly selected atoms.
        # D_pad: Consequentially - we add extra dimensions to the atoms dimensionality axis as well.
        # We do so to ensure "fake atoms" hold information only in these "fake dimensions".
        # As a result, it won't affect calculations such as outer product and matrix inversion.
        amount_of_padding_dimensions = K - torch.min(matrix_num_nonzero)
        K_pad = K + amount_of_padding_dimensions
        D_pad = D + amount_of_padding_dimensions

        # Wd_expanded_dims ~ (D_pad, K)
        # Wd_padded ~ (D_pad, K_pad)
        # The original dictionary with the extra padded dimensions, of both "fake atoms" and extra added
        # dimensions.
        # Wd_expanded_dims: contains a K x K submatix of the original dict, concatenated with empty zeros along
        # the dimension of each original atom, to make real atoms compatible with fake ones.
        # Wd_padded: Contains a K x K submatix of the original dict, expanded with extra zero dimensions,
        # as well as padded with fake atoms. Note that the fake atoms added are simply one hot vecs with information
        # that exists only within the added dimensions.
        # For example (transposed for readability):
        #
        #               dim 1  |  dim 2 | ... | dim D  ||  dim D+1 | ... | dim D_pad
        #   atom 1        x11      x21            xD1        0      0 0 0      0
        #   atom 2        x12      x22            xD2        0      0 0 0      0
        #   ...
        #   atom K        x1K      x2K            xDK        0      0 0 0      0
        #   ======
        #   atom K+1       0        0              0         1        0        0
        #   ...            0        0              0         0        1        0
        #   atom K_pad     0        0              0         0        0        1
        #
        #   W_padded is the matrix above, and Wd_expanded_dims is the subset matrix of K atoms on D+1 dims.
        one_hots = torch.nn.functional.one_hot(torch.arange(D, D_pad), D_pad).to(Wd.device).float().t()
        empty_dims = torch.zeros((amount_of_padding_dimensions, K), device=Wd.device)
        Wd_expanded_dims = torch.cat((Wd, empty_dims), dim=0)
        Wd_padded = torch.cat((Wd_expanded_dims, one_hots), dim=1)

        # D_expanded_dims_batched ~ (B, D_pad, K)
        # D_padded_batched ~ (B, D_pad, K_pad)
        # Next we prepare a dictionary for each patch to choose from.
        # expand operation here repeats the dictionary along the batch dim without copying it,
        # while repeat performs an actual copy.
        D_expanded_dims_batched = Wd_expanded_dims.unsqueeze(dim=0).repeat(B, 1, 1)
        D_padded_batched = Wd_padded.unsqueeze(dim=0).expand(B, *Wd_padded.shape)

        # non_selected_atoms_mask ~ (B, K)
        # First convert the binary mask of selected-unselected atoms to a mask of:
        # zeros for selected atoms
        # fake atom decremental numbers for each unselected atom (Kpad-1 for first, Kpad-1 for second..) per patch
        #
        # For example:
        # -1 -2 -3 ... -K   # Patch 1 haven't selected any atoms (shouldn't happen, just for the sake of example)
        # -1 -2  0 ... -j   # Patch 2 have selected atom 3.
        # 0  -1  0 ... -i   # Patch 3 selected atoms 1 and 3
        #
        # Note: some PyTorch funcs can't deal with negative indexing yet, so instead of using the actual negative
        # indices described above: -1, -2, -3... we use the fake atoms indices, in decreasing order:
        # Kpad-1, Kpad-2, Kpad-3 (e.g: if K=512 and there are 12 fake atoms that's: 523, 522, 521...)
        non_selected_atoms_mask = (K_pad - torch.cumsum(1-matrix_nonzero_inds_mask.long(), dim=1)) * (1-matrix_nonzero_inds_mask.long())

        # base_dict_select_mask ~ (B, K)
        # Then we produce a base mask of:
        # 0 1 2 3 ... K
        # 0 1 2 3 ... K
        # 0 1 2 3 ... K
        # per patch
        base_dict_select_mask = torch.arange(start=0, end=K, device=non_selected_atoms_mask.device).\
            unsqueeze(0). \
            repeat(B, 1)

        # dict_select_mask ~ (B, D_pad, K)
        # Finally we obtain the full mask to select atoms for each patch:
        # 0 1 2 -1 4 5 .. K (atom 3 unselected)
        # 0 1 2 3 4 5 .. K (all atoms selected)
        # 0 1 -1 3 -2 5 .. K (atoms 2 and 4 unselected)
        # and we expand this mask to cover the full atoms dimensionality axis..
        # Note: again, here a negative index -n is actually Kpad-n (since PyTorch crashes for negative indexing
        # sometimes)..
        dict_select_mask = base_dict_select_mask * matrix_nonzero_inds_mask.long() + non_selected_atoms_mask
        dict_select_mask = dict_select_mask.unsqueeze(dim=1).expand(*D_expanded_dims_batched.shape)

        # dict_select_mask ~ (B, D_pad, K)
        # Here we produce the actual dictionary subset matrix.
        # For each patch here the matrix contains only the selected atoms (or fake padded atoms).
        # The gather operation runs along the "atoms" axis, and selects "padded atoms" according the
        # the selection mask we've composed so far (so each patch gains only selected or fake atoms).
        # Note the D_pad usage here: we need the extra padding dimensions because fake atoms
        # are actually the standard basis vectors: e_k+1, e_k+2, ... e_kpad
        # For example:
        #   Assume K=3, D=4, K_pad = 5
        #
        #    Patch #1:  dim0 dim1 dim2 dim3 dim4
        #    Atom 1      x02  x12  x22  0    0
        #    Atom 2       0    0    0   1    0     <-- Atom 2 unselected, replaced with fake Atom 4
        #    Atom 3      x03  x13  x23  0    0
        dict_sampled = torch.gather(dim=2, index=dict_select_mask, input=D_padded_batched)

        # outer_product ~ (B, K, K)
        # Then calculate the outer product of the dictionary subset -
        # Every entry here is either:
        # - a correlation of 2 atoms
        # - zero, if this is a correlation of 2 atoms and at least one is fake
        # - one, if this is a correlation of a fake atom with itself.
        dict_sampled_t = dict_sampled.transpose(1, 2)
        outer_product = torch.bmm(dict_sampled_t, dict_sampled)

        # outer_product ~ (B, K, K)
        # Add a small perturbation to maintain numerical stability and avoid singular non-invertible matrices
        # (fake atom dimensions don't get perturbed)
        I = torch.eye(K, dtype=outer_product.dtype, device=outer_product.device).unsqueeze(0).repeat(B, 1, 1)
        expanded_inds_mask = matrix_nonzero_inds_mask.unsqueeze(-1).expand_as(I).float()
        I_perturb = I * expanded_inds_mask * peturb_eps

        # Invert the perturbed outer product to obtain the gradient weights.
        # We wrap up the calculation by initializing the beta entries with the weighted gradients.
        matrix_dict_grad_weights = (outer_product + I_perturb).contiguous()
        matrix_dict_grad_weights_inv = torch.inverse(matrix_dict_grad_weights)
        matrix_grads = matrix_grad_output.t() * matrix_nonzero_inds_mask.float()
        matrix_weighted_dict_grads = matrix_dict_grad_weights_inv.bmm(matrix_grads.unsqueeze(-1))
        matrix_beta[patch_index_list] = matrix_weighted_dict_grads.squeeze(-1)

        return matrix_beta

    @staticmethod
    def backward(ctx, grad_output, num_steps_placeholder, debug=False):
        """
        Deriving gradients of the dictionary with respect to the selected sparse code

        Dimension notation:
            B - Number of samples. Usually number of patches in image times batch size
            K - Number of atoms in dictionary
            d - Dimensionality of atoms in dictionary
            k - Number of non-zero coefficients in the selected sparse code

        Variable explanation:
        X: Input - Signal to find sparse coding against. Dimensions: d X B
        Wd: Dictionary - Tensor of atoms we want to get a sparse linear combination of. Dimensions: d X K
        Z: Selected sparse code (results of FISTA). Dimensions: K x B
        grad_dictionary: Gradients recieved for the dictionary in respect to down-stream task (reconstruction, etc.)
        """
        start = time()
        X, Wd, Z = ctx.saved_tensors
        grad_input = grad_dictionary = alpha = tolerance = max_steps = debug = sample_gradients = simd_gradients = None

        # Epsilon values, used within the algorithm for thresholding zero float values and petrubrations for
        # numerical stability
        eps = np.finfo(float).eps * 10
        peturb_eps = eps * 10000

        # Skip if we don't need to backward through the dictionary
        if not ctx.needs_input_grad[1]:
            return grad_input, grad_dictionary, alpha, tolerance, max_steps, debug, sample_gradients, simd_gradients

        betas = torch.zeros(grad_output.size()).type(grad_output.type()).t().to(grad_output.device)
        patch_times = []
        beta_times = []
        inversion_times = []
        weighing_times = []
        if debug:
            print('Doing backward, sampling gradients: {}'.format(ctx.sample_gradients))

        # Sample patches to backward on or do all of them
        patch_index_list = list(range(Z.size()[1]))
        if ctx.sample_gradients:
            patch_indices = np.random.choice(patch_index_list, min(ctx.sample_gradients, len(patch_index_list)) , False)
            patch_index_list = list(patch_indices)

        done = 0

        if not ctx.use_backwards_simd:
            # We gather weight coefficients for every gradient for every patch
            for patch_index in patch_index_list:
                patch_time = time()
                nonzero_inds = Z[:, patch_index].nonzero().squeeze(1)  # Size should be [k]
                num_nonzero = nonzero_inds.nelement()

                # If sparse code selected nothing, gradient should be 0
                if num_nonzero == 0:
                    continue

                # Calculate beta left side
                beta_start = time()
                sampled_dict = Wd[:, nonzero_inds]
                dict_grads_weights = sampled_dict.t().mm(sampled_dict)
                dict_grads_weights += + peturb_eps * torch.eye(len(dict_grads_weights)).to(dict_grads_weights.device)  # Used to make DtD

                inversion_start_time = time()
                try:
                    inverted_grad_weights = torch.inverse(dict_grads_weights)
                except Exception as e:
                    print('FAILED inverting gradients with gradient size {}. Error msg: {}'.format(dict_grads_weights.size(), e))
                    continue

                beta_end_time = time()
                inversion_times.append(time() - inversion_start_time)
                beta_times.append(beta_end_time-beta_start)

                weighting_start = time()
                sampled_dict_grads = grad_output[nonzero_inds, patch_index]
                weighted_dict_grads = inverted_grad_weights.mm(sampled_dict_grads.unsqueeze(1))
                weighing_times.append(time()-weighting_start)

                # Gather gradient weight coefficients per patch
                betas[patch_index, nonzero_inds] += weighted_dict_grads.t().squeeze()
                done += 1
                patch_times.append(time()-patch_time)

        # @TODO: Yiftach Ginger - This block demonstrates the SIMD implementation
        if ctx.use_backwards_simd:
            # ---
            matrix_start_time = time()
            matrix_betas = FistaFunction.backward_simd(Z, Wd, grad_output, patch_index_list)
            matrix_end_time = time() - matrix_start_time
            if debug:
                print('Done simd backwards in {} seconds'.format(matrix_end_time))
            betas = matrix_betas
            # assert(torch.sum(matrix_betas - betas) < peturb_eps)
            # ---


        weighted_quantization = -Wd.mm(betas.t().mm(Z.t()))
        weighted_quantized_difference = (X - Wd.mm(Z)).mm(betas)
        grad_dictionary = weighted_quantization + weighted_quantized_difference

        if debug:
            print('Backwarded through {} / {} samples'.format(done, ctx.sample_gradients))
            print('Done patch in {} seconds'.format(np.mean(patch_times)))
            print('Done beta in {} seconds'.format(np.mean(beta_times)))
            print('Done inversion_times in {} seconds'.format(np.mean(inversion_times)))
            print('Done weighing_times in {} seconds'.format(np.mean(weighing_times)))
            print('Done backward in {} seconds'.format(time() - start))
        return grad_input, grad_dictionary, alpha, tolerance, max_steps, debug, sample_gradients, simd_gradients


def FISTA(X, Wd, alpha, tolerance, max_steps=-1, debug=False):
    """
    Dimension notation:
        B - Number of samples. Usually number of patches in image times batch size
        K - Number of atoms in dictionary
        d - Dimensionality of atoms in dictionary


    :param X: Input - Signal to find sparse coding against. Dimenions: d x B
    :param Wd: Dictionary - Tensor of atoms we want to get a sparse linear combination of. Dimensions: d x K
    :param alpha:  Float. Sparsity weight
    :param tolerance: Float. Stop optimization once the improvement drops below this term.
    :param max_steps: Int. Stops optimizaiton after max_steps iterations. Ignored if < 0.
    :param debug: Bool. Output debug message if True.

    :return: Z: linear coefficients for Sparse Code solution. Dimensions: K x B
    """
    if debug:
        print('Doing FISTA with alpha={}'.format(alpha))
    L = get_largest_eigenvalue(Wd.t().mm(Wd))
    prev_Z = torch.zeros((Wd.size()[1], X.size()[1]), dtype=X.dtype).to(Wd.device)
    Y = prev_Z
    momentum = torch.ones(1).to(Wd.device)

    max_distance = 2 * tolerance
    step = 0
    while max_distance > tolerance or (step == 1 and torch.isnan(max_distance).item()):
        # Update sparse code
        cur_Z = fista_step(L, Wd, X, alpha, Y)
        Y, momentum = fista_momentum(cur_Z, prev_Z, momentum)
        # next_Z = cur_Z

        # calculate distance for tolerance
        distance_nominator = torch.norm(Wd.mm(prev_Z - cur_Z), p=2, dim=0)
        distance_denominator = torch.norm(Wd.mm(prev_Z),p=2, dim=0)
        distance = distance_nominator / (distance_denominator + np.finfo(float).eps)
        max_distance = distance.max()

        prev_Z = cur_Z

        step += 1
        if debug and step % 1 == 0:
            print('Step {}, code improvement: {}, below tolerance: {}.'.format(step, max_distance, (distance < tolerance).float().mean().item()))

        if step >= max_steps > 0:
            break

    return cur_Z, step


def fista_momentum(cur_Z, prev_Z, momentum):
    """
    Calculates a linear combination of the last two sparse codings with a momentum term

    :param cur_Z: Sparse code found in current step
    :param prev_Z: Sparse code found in previous step
    :param momentum: float. Momentum term.
    :return: Updated sparse code to be used in next step.
    """
    next_momentum = (1 + torch.sqrt(1+4*(momentum**2)))/2
    momentum_ratio = (momentum - 1) / next_momentum

    pushed_Z = (cur_Z - prev_Z) * momentum_ratio
    next_Z = cur_Z + pushed_Z

    return next_Z, next_momentum


def fista_step(L, Wd, X, alpha, last_Z):
    """
        Calculates the next sparse code for the FISTA algorithm

        Dimension notation:
        B - Number of samples. Usually number of patches in image times batch size
        K - Number of atoms in dictionary
        d - Dimensionality of atoms in dictionary

        :param X: Input - Signal to find sparse coding against. Dimensions: d X B
        :param Wd: Dictionary - Tensor of atoms we want to get a sparse linear combination of. Dimensions: d X K
        :param alpha:  Float. Sparsity weight
        :param L: Float. Largest eigenvalue in Wd
        :param last_Z: Sparse code from previous step. Dimensions: K x B

        :return: Z: linear coefficients for Sparse Code solution. Dimensions: K x B
        """

    quantization_distance = Wd.mm(last_Z) - X.to(Wd.device)
    normalized_dictionary = Wd.t() / L
    normalized_quantization_projection = normalized_dictionary.mm(quantization_distance)
    cur_Z = last_Z - normalized_quantization_projection
    cur_Z = shrink_function(cur_Z, alpha / L)
    return cur_Z


def test_fista_gradient():
    from torch.autograd import gradcheck
    from functools import partial

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (torch.randn(8, 1, dtype=torch.double, requires_grad=True), torch.randn((8, 512), dtype=torch.double, requires_grad=True))
    # input = (torch.randn(8, 30, dtype=torch.double, requires_grad=True).cuda(0), torch.randn((8, 64), dtype=torch.double, requires_grad=True).cuda(0))
    test = gradcheck(FistaFunction.apply, input, eps=1e-6, atol=1e-4)
    print(test)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    num_samples = 4096*128
    dictionary_size = 512
    embedding_size = 64
    X = torch.randn(embedding_size, num_samples).cuda()
    Wd = torch.randn(embedding_size, dictionary_size).cuda()
    alpha = 3
    from time import time
    t = time()
    Z = FISTA(X, Wd, alpha, 0.01, debug=True)
    print(time()-t)

    # test_fista_gradient()