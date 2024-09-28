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

def _prod(l):
    out = 1
    for x in l: out *= x
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

def sparse_mm(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """
    Perform batched sparse matrix multiplication A @ B where A is a sparse COO tensor
    almost all of whose rows are 0.
    
    Args:
    A (t.Tensor): Batched sparse COO tensor of shape [..., f, d]
    B (t.Tensor): Batched dense tensor of shape [..., d, f]
    
    Returns:
    t.Tensor: Batched sparse COO tensor result of A @ B
    """
    assert A.is_sparse and A.layout == t.sparse_coo
    assert A.dim() >= 2 and B.dim() >= 2
    assert A.shape[-1] == B.shape[-2]
    assert A.device == B.device, "Both tensors must be on the same device"
    
    device = A.device
    batch_dims_A, f, d = A.shape[:-2], A.shape[-2], A.shape[-1]
    batch_dims_B, d, f_out = B.shape[:-2], B.shape[-2], B.shape[-1]
    
    assert batch_dims_A == batch_dims_B, "Batch dimensions must match"
    
    # Extract indices and values from sparse tensor A
    indices = A._indices()
    values = A._values()
    
    # Compute the number of elements in each batch
    batch_size = t.prod(t.tensor(batch_dims_A)).item()
    
    # Flatten B for batched multiplication
    B_flat = B.reshape(-1, d, f_out)
    
    # Compute batch indices
    batch_indices = t.tensor([t.prod(t.tensor(batch_dims_A[i+1:])).item() for i in range(len(batch_dims_A))], device=device)
    batch_indices = (indices[:-2] * batch_indices.view(-1, 1)).sum(dim=0).long()
    
    # Perform multiplication for non-zero elements
    result_values = B_flat[batch_indices, indices[-1].long()] * values.unsqueeze(-1)
    
    # Create new indices for the result
    row_indices = indices[-2].unsqueeze(-1).expand(-1, f_out).flatten()
    col_indices = t.arange(f_out, device=device).expand(indices.size(1), -1).flatten()
    new_batch_indices = batch_indices.repeat_interleave(f_out)
    
    result_indices = t.stack([*[new_batch_indices // t.prod(t.tensor(batch_dims_A[i+1:])).item() % batch_dims_A[i] 
                                for i in range(len(batch_dims_A))], 
                              row_indices, 
                              col_indices]).long()
    
    # Create the sparse result tensor
    result = t.sparse_coo_tensor(
        result_indices,
        result_values.flatten(),
        size=(*batch_dims_A, f, f_out),
        device=device
    ).coalesce()
    
    return result

def jank_multiply(A, B):
    indices = []
    values = []
    B_indices = B.indices().T.tolist()
    for idx in A.indices().T.tolist():
        if idx in B_indices:
            indices.append(idx)
            values.append(A[tuple(idx)] * B[tuple(idx)])
    indices = t.tensor(indices, device=A.device).T
    values = t.tensor(values, device=A.device)
    return t.sparse_coo_tensor(indices, values, size=A.shape)

def sparsely_batched_outer_prod(A, B):
    """
    A: fully sparse [..., a|]
    B: fully dense [|a, ...]
    Outputs: A*B viewed as shape [... a | ...] 
    """
    assert A.is_sparse and A.layout == t.sparse_coo
    assert A.shape[-1] == B.shape[0]

    A = A.coalesce()
    indices = A.indices() # [A.dim(), nnz]
    B_values = B[indices[-1,:]] # [nnz, ...]
    values = (
        A.values().view(-1, *[1 for _ in range(B.dim() - 1)])
    ) * B_values

    return t.sparse_coo_tensor(indices, values, size=(*A.shape[:-1], *B.shape))

def doubly_batched_inner_prod(A, B):
    """
    A: [..., | d]
    B: [..., | d]
    Returns: all pairwise inner products, fully sparse [..., ...]
    """
    assert A.shape[-1] == B.shape[-1]

    # if A.dim() == 1:
    #     B = B.coalesce()
    #     indices = B.indices()
    #     values = A @ B.values().T
    # elif B.dim() == 1:
    #     A = A.coalesce()
    #     indices = A.indices()
    #     values = A.values() @ B
    # else:
    A, B = A.coalesce(), B.coalesce()
    A_idxs, B_idxs = A.indices(), B.indices()
    indices = t.concat(
        [
            A_idxs.unsqueeze(2).expand(-1, -1, B_idxs.shape[1]),
            B_idxs.unsqueeze(1).expand(-1, A_idxs.shape[1], -1),
        ],
        dim=0
    )
    indices = indices.reshape(indices.shape[0], -1)

    values = (A.values() @ B.values().T).flatten()

    return t.sparse_coo_tensor(indices, values, size=(*A.shape[:-1], *B.shape[:-1]))


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


    # test sparse_mm
    W = t.randn(100, 50).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
    b, s = 5, 10
    idxs = t.stack(
        [
            t.randint(0, b, (15,)),
            t.randint(0, s, (15,)),
            t.randint(0, 100, (15,))
        ],
        dim=1
    ).to(W.device)
    idxs = t.unique(idxs, dim=0)
    W_out = sparsely_expand(W, idxs, b, s).to_dense()

    for i in range(b):
        for j in range(s):
            for k in range(W.shape[0]):
                if [i,j,k] in idxs.tolist():
                    assert t.all(W_out[i, j, :, :, k, :] == W[k].expand(b, s, -1))
                else:
                    assert t.all(W_out[i, j, :, :, k, :] == 0)

    A = t.randn(3, 5, 2, 100, 50).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
    B = t.randn(3, 5, 2, 50, 100).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
    A_sparse = A.to_sparse_coo()
    result = sparse_mm(A_sparse, B)
    assert t.allclose(A @ B, result.to_dense(), atol=1e-5)


# test jank_multiply
A = t.randn(10, 3, 5).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
B = t.randn(10, 3, 5).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
A[t.rand_like(A) > 0.5] = 0
B[t.rand_like(B) > 0.5] = 0
A, B = A.to_sparse_coo(), B.to_sparse_coo()

result = jank_multiply(A, B)
assert t.allclose(result.to_dense(), A.to_dense() * B.to_dense())

# test sparsely_batched_outer_prod
A = t.randn(10, 3, 5).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
A[t.rand_like(A) > 0.5] = 0
A = A.to_sparse_coo()
B = t.randn(5, 30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))

result = sparsely_batched_outer_prod(A, B)
assert t.allclose(result.to_dense(), A.to_dense().unsqueeze(-1) * B)

# test doubly_batched_inner_prod
A = t.randn(10, 3, 5, 30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
B = t.randn(8, 9, 30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
A[t.rand_like(A) > 0.5] = 0
B[t.rand_like(B) > 0.5] = 0

result_dense = (A.view(10, 3, 5, 1, 1, 30) * B.view(1, 1, 8, 9, 30)).sum(dim=-1)

A = A.to_sparse(sparse_dim=3)
B = B.to_sparse(sparse_dim=2)
result_sparse = doubly_batched_inner_prod(A, B)

assert t.allclose(result_sparse.to_dense(), result_dense, atol=1e-4)

# A = t.randn(30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
# B = t.randn(8, 9, 30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
# B[t.rand_like(B) > 0.5] = 0

# result_dense = (A * B).sum(dim=-1)

# B = B.to_sparse(sparse_dim=2)
# result_sparse = doubly_batched_inner_prod(A, B)

# assert t.allclose(result_sparse.to_dense(), result_dense, atol=1e-4)

# A = t.randn(10, 3, 5, 30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
# B = t.randn(30).to(t.device("cuda") if t.cuda.is_available() else t.device("cpu"))
# A[t.rand_like(A) > 0.5] = 0

# result_dense = (A * B).sum(dim=-1)

# A = A.to_sparse(sparse_dim=3)
# result_sparse = doubly_batched_inner_prod(A, B)

# assert t.allclose(result_sparse.to_dense(), result_dense, atol=1e-4)