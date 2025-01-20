import torch as t

"""
Utils for handling sparse COO tensors
"""

def _flatten_index(idxs, shape):
    """
    index : a tensor of shape [n, len(shape)]
    shape : a shape
    return a tensor of shape [n] where each element is the flattened index
    """
    idxs = idxs.t()
    # get strides from shape
    strides = [1]
    for i in range(len(shape)-1, 0, -1):
        strides.append(strides[-1]*shape[i])
    strides = list(reversed(strides))
    strides = t.tensor(strides).to(idxs.device)
    # flatten index
    return (idxs * strides).sum(dim=1).unsqueeze(0)

def _prod(li):
    out = 1
    for x in li: 
        out *= x
    return out

def sparse_flatten(x):
    x = x.coalesce()
    return t.sparse_coo_tensor(
        _flatten_index(x.indices(), x.shape),
        x.values(),
        (_prod(x.shape),)
    )

def _reshape_index(index, shape):
    """
    index : a tensor of shape [n]
    shape : a shape
    return a tensor of shape [n, len(shape)] where each element is the reshaped index
    """
    multi_index = []
    for dim in reversed(shape):
        multi_index.append(index % dim)
        index //= dim
    multi_index.reverse()
    return t.stack(multi_index, dim=-1)

def sparse_reshape(x, shape):
    """
    x : a sparse COO tensor
    shape : a shape
    return x reshaped to shape
    """
    # first flatten x
    x = sparse_flatten(x).coalesce()
    new_indices = _reshape_index(x.indices()[0], shape)
    return t.sparse_coo_tensor(new_indices.t(), x.values(), shape).coalesce()

def sparse_mean(x, dim):
    if isinstance(dim, int):
        return x.sum(dim=dim) / x.shape[dim]
    else:
        return x.sum(dim=dim) / _prod(x.shape[d] for d in dim)

def sparse_repeat(x, repeat_sizes):
    """
    Repeats a sparse COO tensor along specified dimensions.
    
    Args:
    x (t.Tensor): Input sparse COO tensor
    *repeat_sizes: The number of times to repeat this tensor along each dimension
    
    Returns:
    t.Tensor: A new sparse COO tensor with repeated values
    """
    x = x.coalesce()
    old_shape = x.shape
    old_indices = x.indices()
    old_values = x.values()
    
    new_shape = tuple(s * r for s, r in zip(old_shape, repeat_sizes))
    total_repeat = _prod(repeat_sizes)
    
    new_indices = []
    for dim, (size, repeat) in enumerate(zip(old_shape, repeat_sizes)):
        indices = old_indices[dim].repeat(total_repeat)
        
        offsets = t.arange(repeat, device=x.device)
        for r in repeat_sizes[dim+1:]:
            offsets = offsets.unsqueeze(-1).expand(*offsets.shape, r).reshape(-1)
        offsets = offsets.repeat_interleave(old_indices.shape[1])
        
        if offsets.size(0) < indices.size(0):
            offsets = offsets.repeat(indices.size(0) // offsets.size(0))
        
        new_indices.append(indices + offsets * size)
    
    new_indices = t.stack(new_indices)
    new_values = old_values.repeat(total_repeat)
    
    return t.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()

def sparsely_expand(W, idxs, b, s):
    """
    Given: a tensor W of shape [f, d] and a tensor of indices idxs of shape [n, 3]
    such that idxs[:, 0] < b, idxs[:, 1] < s, idxs[:, 2] < f.
    Return: an expanded sparse COO tensor W_out of shape [b, s, b, s, f, d] satisfying:
      W_out[i, j, :, :, k, :] == W[k].expand(b, s, d) if idxs contains [i, j, k]
      W_out[i, j, :, :, k, :] == 0 otherwise
    """

    f, d = W.shape
    device = W.device
    n = idxs.shape[0]

    # Create expanded indices
    i, j, k = idxs.t()
    i_rep = t.arange(b, device=device).view(1, -1, 1, 1).expand(n, -1, s, d)
    j_rep = t.arange(s, device=device).view(1, 1, -1, 1).expand(n, b, -1, d)
    d_idx = t.arange(d, device=device).view(1, 1, 1, -1).expand(n, b, s, -1)

    # Create the indices for the sparse tensor
    indices = t.stack([
        i.view(-1, 1, 1, 1).expand(-1, b, s, d),
        j.view(-1, 1, 1, 1).expand(-1, b, s, d),
        i_rep,
        j_rep,
        k.view(-1, 1, 1, 1).expand(-1, b, s, d),
        d_idx
    ], dim=0).permute(1, 2, 3, 4, 0).reshape(-1, 6).t()

    # Create the values for the sparse tensor
    values = W[k].view(-1, 1, 1, d).expand(-1, b, s, -1).reshape(-1)

    # Create the sparse COO tensor
    return t.sparse_coo_tensor(
        indices,
        values,
        size=(b, s, b, s, f, d),
        device=device
    )

def sparse_prod(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """
    If A is a sparse COO tensor of shape [..., f] and B is a dense tensor of shape [f, d],
    return a sparse COO tensor of shape [..., f, d] equal to
    A.unsqueeze(-1) * B.
    """
    d = B.shape[-1]

    indices = A.indices()
    dims, nonzeros = indices.shape

    B_values = B[A.indices()[-1,:]]
    B_values = B_values.flatten()
    A_values = A.values()
    A_values = A_values.unsqueeze(-1).expand(-1, d).flatten()
    values = A_values * B_values

    indices = indices.unsqueeze(-1).expand(*indices.shape, d).reshape(dims, -1)
    indices = t.cat([indices, t.arange(d, device=A.device).repeat(nonzeros).unsqueeze(0)], dim=0)

    return t.sparse_coo_tensor(indices, values, (*A.shape, d))

if __name__ == "__main__":
    x = t.randn(50, 50, 50)
    x = x.to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))

    x_sparse = x.to_sparse_coo()

    # test sparse_flatten
    assert t.allclose(x.flatten(), sparse_flatten(x_sparse).to_dense())

    # test sparse_mean
    for dim in range(3):
        assert t.allclose(x.mean(dim=dim), sparse_mean(x_sparse, dim).to_dense(), atol=1e-4)

    # test sparse_reshape
    shape = (5, 10, 2500)
    x_reshaped = x.view(shape)
    x_sparse_reshaped = sparse_reshape(x_sparse, shape)
    assert t.allclose(x_reshaped, x_sparse_reshaped.to_dense())

    # test sparse repeat
    repeat_sizes = (2, 1, 4)
    x_repeated = x.repeat(*repeat_sizes)
    x_sparse_repeated = sparse_repeat(x_sparse, repeat_sizes)
    assert t.allclose(x_repeated, x_sparse_repeated.to_dense())

    # test sparse_prod
    b = 5
    s = 3
    f = 50
    d = 30
    A = t.randn(b, s, f).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
    A[t.rand_like(A) > 0.5] = 0
    B = t.randn(f, d).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
    A_sparse = A.to_sparse_coo()
    result = sparse_prod(A_sparse, B)
    assert t.allclose(result.to_dense(), A.unsqueeze(-1) * B, atol=1e-5)