from __future__ import annotations
import torch as t
from torchtyping import TensorType
from dictionary_learning import AutoEncoder

class SparseAct():
    def __init__(
            self, 
            dense_act: TensorType["batch_size", "n_ctx", "d_model"] = None, 
            dictionary: AutoEncoder = None,
            sparse_act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None, 
            residual: TensorType["batch_size", "n_ctx", "d_model"] = None,
            ) -> None:

            def is_dense_init(dense_act, dictionary, sparse_act, residual):
                """Check if initialization is for dense activation mode."""
                return dense_act is not None and dictionary is not None and sparse_act is None and residual is None

            def is_sparse_init(dense_act, dictionary, sparse_act, residual):
                """Check if initialization is for sparse activation mode."""
                return dense_act is None and dictionary is None and sparse_act is not None and residual is not None

            if is_dense_init(dense_act, dictionary, sparse_act, residual):
                self.sparse_act = dictionary.encode(dense_act)
                reconstructed_act = dictionary.decode(self.sparse_act)
                self.residual = dense_act - reconstructed_act
            elif is_sparse_init(dense_act, dictionary, sparse_act, residual):
                self.sparse_act = sparse_act
                self.residual = residual
            else:
                raise ValueError("Please initialize SparseAct with either (dense_act and dictionary) XOR (sparse_act and residual) arguments.")

    def __mul__(self, other: SparseAct) -> SparseAct:
        sparse_result = self.sparse_act * other.sparse_act
        residual_result = self.residual * other.residual
        return SparseAct(sparse_act=sparse_result, residual=residual_result)
    
    def __minus__(self, other: SparseAct) -> SparseAct:
        sparse_result = self.sparse_act - other.sparse_act
        residual_result = self.residual - other.residual
        return SparseAct(sparse_act=sparse_result, residual=residual_result)
    
    def __neg__(self) -> SparseAct:
        sparse_result = -self.sparse_act
        residual_result = -self.residual
        return SparseAct(sparse_act=sparse_result, residual=residual_result)
    
    def value(self):
        self.sparse_act = self.sparse_act.value
        self.residual = self.residual.value
        return SparseAct(sparse_act=self.sparse_act, residual=self.residual)
    
    def detach(self):
        self.sparse_act = self.sparse_act.detach()
        self.residual = self.residual.detach()
        return SparseAct(sparse_act=self.sparse_act, residual=self.residual)


if __name__ == "__main__":
    # Initialize SparseAct
    batch_size = 2
    n_ctx = 16
    d_model = 512
    d_dictionary = d_model * 64

    dense_A = t.rand((batch_size, n_ctx, d_model))
    dense_B = t.rand((batch_size, n_ctx, d_model))
    dictionary_layer = 0
    ae = AutoEncoder(d_model, d_dictionary)
    ae.load_state_dict(t.load(f'/share/projects/dictionary_circuits/autoencoders/pythia-70m-deduped/mlp_out_layer{dictionary_layer}/5_32768/ae.pt'))

    sa_A = SparseAct(dense_act=dense_A, dictionary=ae)
    sa_B = SparseAct(dense_act=dense_B, dictionary=ae)

    # Elementwise Multiplication
    sa_elementwise = sa_A @ sa_B
    t.allclose(sa_elementwise.sparse_act, sa_A.sparse_act * sa_B.sparse_act)
    print("Elementwise Multiplication Test Passed")