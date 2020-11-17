import torch
from sklearn.decomposition import DictionaryLearning


def get_largest_eigenvalue(X):
    eigs = torch.eig(X, eigenvectors=False).eigenvalues
    max_eign = eigs.max(dim=0)
    return max_eign.values[0]


def shrink_function(Z, cutoff):
    cutted = shrink1(Z, cutoff)
    maxed = shrink2(Z, cutted)
    signed = shrink3(Z, maxed)
    return signed


def shrink3(Z, maxed):
    signed = maxed * torch.sign(Z)
    return signed


def shrink2(Z, cutted):
    maxed = torch.max(cutted, torch.zeros(Z.size(), dtype=Z.dtype).cuda(3))
    return maxed


def shrink1(Z, cutoff):
    cutted = torch.abs(Z) - cutoff
    return cutted


def reconstruction_distance(D, cur_Z, last_Z):
    distance = torch.norm(D.mm(last_Z - cur_Z), p=2, dim=0) / torch.norm(D.mm(last_Z), p=2, dim=0)
    max_distance = distance.max()
    return distance, max_distance


def OMP(X, D, K, tolerance, debug=False):
    Dt = D.t()
    Dpinv = torch.pinverse(D)
    r = X
    I = []
    stopping = False
    last_sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype)#.cuda(3)
    sparse_code = torch.zeros((D.size()[1], X.size()[1]), dtype=X.dtype)#.cuda(3)

    step = 0
    while not stopping:
        k_hat = torch.argmax(Dt.mm(r), 0)
        I.append(k_hat)
        sparse_code = Dpinv.mm(X) # Should be: (torch.pinverse(D[:,I])*X).sum(0)
        r = X - D.mm(sparse_code)

        distance, max_distance = reconstruction_distance(D, sparse_code, last_sparse_code)
        stopping = len(I) >= K or max_distance < tolerance
        last_sparse_code = sparse_code

        if debug and step % 1 == 0:
            print('Step {}, code improvement: {}, below tolerance: {}'.format(step, max_distance, (distance < tolerance).float().mean().item()))

        step += 1

    return sparse_code


def _update_logical(logical, to_add):
    running_idx = torch.arange(to_add.shape[0], device=to_add.device)
    logical[running_idx, to_add] = 1


def Batch_OMP(data, dictionary, max_nonzero, tolerance=1e-7, debug=False):
    """
    for details on variable names, see
    https://sparse-plex.readthedocs.io/en/latest/book/pursuit/omp/batch_omp.html
    or the original paper
    http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

    NOTE - the implementation below works on transposed versions of the input signal to make the batch size the first
           coordinate, which is how pytorch expects the data..
    """
    vector_dim, batch_size = data.size()
    dictionary_t = dictionary.t()
    G = dictionary_t.mm(dictionary)  # this is a Gram matrix
    eps = torch.norm(data, dim=0)  # the residual, initalized as the L2 norm of the signal
    h_bar = dictionary_t.mm(data).t()  # initial correlation vector, transposed to make batch_size the first dimension

    # note - below this line we no longer use "data" or "dictionary"

    h = h_bar
    x = torch.zeros_like(h_bar)  # the resulting sparse code
    L = torch.ones(batch_size, 1, 1, device=h.device)  # Contains the progressive Cholesky of G in selected indices
    I = torch.ones(batch_size, 0, device=h.device).long()
    I_logic = torch.zeros_like(h_bar).bool()  # used to zero our elements is h before argmax
    delta = torch.zeros(batch_size, device=h.device)  # to track errors

    k = 0
    while k < max_nonzero and eps.max() > tolerance:
        k += 1
        # use "I_logic" to make sure we do not select same index twice
        index = (h*(~I_logic).float()).abs().argmax(dim=1)  # todo - can we use "I" rather than "I_logic"
        _update_logical(I_logic, index)
        batch_idx = torch.arange(batch_size, device=G.device)
        expanded_batch_idx = batch_idx.unsqueeze(0).expand(k, batch_size).t()

        if k > 1:  # Cholesky update
            # Following line is equivalent to:
            #   G_stack = torch.stack([G[I[i, :], index[i]] for i in range(batch_size)], dim=0).view(batch_size, k-1, 1)
            G_stack = G[I[batch_idx, :], index[expanded_batch_idx[...,:-1]]].view(batch_size, k-1, 1)
            w = torch.triangular_solve(G_stack, L, upper=False, ).solution.view(-1, 1, k-1)
            w_corner = torch.sqrt(1-(w**2).sum(dim=2, keepdim=True))  # <- L corner element: sqrt(1- w.t().mm(w))

            # do concatenation into the new Cholesky: L <- [[L, 0], [w, w_corner]]
            k_zeros = torch.zeros(batch_size, k-1, 1, device=h.device)
            L = torch.cat((
                torch.cat((L, k_zeros), dim=2),
                torch.cat((w, w_corner), dim=2),
            ), dim=1)

        # update non-zero indices
        I = torch.cat([I, index.unsqueeze(1)], dim=1)

        # x = solve L
        # The following line is equivalent to:
        #   h_stack = torch.stack([h_bar[i, I[i, :]] for i in range(batch_size)]).unsqueeze(2)
        h_stack = h_bar[expanded_batch_idx, I[batch_idx, :]].view(batch_size, k, 1)
        x_stack = torch.cholesky_solve(h_stack, L)

        # de-stack x into the non-zero elements
        # The following line is equivalent to:
        #   for i in range(batch_size):
        #       x[i:i+1, I[i, :]] = x_stack[i, :].t()
        x[batch_idx.unsqueeze(1), I[batch_idx]] = x_stack[batch_idx].squeeze(-1)

        # beta = G_I * x_I
        # The following line is equivalent to:
        # beta = torch.cat([x[i:i+1, I[i, :]].mm(G[I[i, :], :]) for i in range(batch_size)], dim=0)
        beta = x[batch_idx.unsqueeze(1), I[batch_idx]].unsqueeze(1).bmm(G[I[batch_idx], :]).squeeze(1)

        h = h_bar - beta

        # update residual
        new_delta = (x * beta).sum(dim=1)
        eps += delta-new_delta
        delta = new_delta

        if debug and k % 1 == 0:
            print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(k, eps.max(), (eps < tolerance).float().mean().item()))

    return x.t()  # transpose since sparse codes should be used as D * x


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    torch.manual_seed(0)
    use_gpu = torch.cuda.device_count() > 0
    device = 'cuda' if use_gpu else 'cpu'

    num_nonzeros = 4
    num_samples = int(1e4)
    num_atoms = 512
    embedding_size = 64

    Wd = torch.randn(embedding_size, num_atoms)
    Wd = torch.nn.functional.normalize(Wd, dim=0).to(device)

    codes = []
    for i in tqdm(range(num_samples), desc='generating codes... '):
        tmp = torch.zeros(num_atoms).to(device)
        tmp[torch.randperm(num_atoms)[:num_nonzeros]] = 0.5 * torch.rand(num_nonzeros).to(device) + 0.5
        codes.append(tmp)
    codes = torch.stack(codes, dim=1)
    X = Wd.mm(codes)
    # X += torch.randn(X.size()) / 100  # add noise
    # X = torch.nn.functional.normalize(X, dim=0)  # normalize signal

    if use_gpu:  # warm start?
        print('doing warm start...')
        Batch_OMP(X[:, :min(num_nonzeros, 1000)], Wd, num_nonzeros)

    tic = time.time()
    Z2 = Batch_OMP(X, Wd, num_nonzeros, debug=True)
    Z2_time = time.time() - tic
    print(f'Z2, {torch.isclose(codes, Z2, rtol=1e-03, atol=1e-05).float().mean()}, time/sample={1e6*Z2_time/num_samples/num_nonzeros:.4f}usec')
    pass


