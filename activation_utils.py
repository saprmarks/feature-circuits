from __future__ import annotations
import torch as t
from torchtyping import TensorType
from dictionary_learning import AutoEncoder

class SparseAct():
    def __init__(
            self, 
            act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None, 
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            # dense_act: TensorType["batch_size", "n_ctx", "d_model"] = None, 
            # dictionary: AutoEncoder = None,
            ) -> None:

            # def is_dense_init(dense_act, dictionary, act, res):
            #     """Check if initialization is for dense activation mode."""
            #     return dense_act is not None and dictionary is not None and act is None and res is None

            # def is_sparse_init(dense_act, dictionary, act, res):
            #     """Check if initialization is for sparse activation mode."""
            #     return dense_act is None and dictionary is None and act is not None and res is not None

            # if is_dense_init(dense_act, dictionary, act, res):
            #     self.act = dictionary.encode(dense_act)
            #     reconstructed_act = dictionary.decode(self.act)
            #     self.res = dense_act - reconstructed_act
            # elif is_sparse_init(dense_act, dictionary, act, res):
            self.act = act
            self.res = res
            # else:
            #     raise ValueError("Please initialize SparseAct with either (dense_act and dictionary) XOR (act and res) arguments.")

    def __mul__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            # Handle SparseAct * SparseAct
            act_result = self.act * other.act
            res_result = self.res * other.res
        elif isinstance(other, (float, int)):
            # Handle SparseAct * float/int
            act_result = self.act * other
            res_result = self.res * other
        else:
            raise TypeError(f"Unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")
        return SparseAct(act=act_result, res=res_result)

    def __rmul__(self, other) -> 'SparseAct':
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)
    
    def __add__(self, other: SparseAct) -> SparseAct:
        sparse_result = self.act + other.act
        res_result = self.res + other.res
        return SparseAct(act=sparse_result, res=res_result)
    
    def __sub__(self, other: SparseAct) -> SparseAct:
        sparse_result = self.act - other.act
        res_result = self.res - other.res
        return SparseAct(act=sparse_result, res=res_result)
    
    def __neg__(self) -> SparseAct:
        sparse_result = -self.act
        res_result = -self.res
        return SparseAct(act=sparse_result, res=res_result)
    
    def value(self):
        self.act = self.act.value
        self.res = self.res.value
        return SparseAct(act=self.act, res=self.res)
    
    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseAct(act=self.act, res=self.res)


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