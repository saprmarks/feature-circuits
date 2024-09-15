from __future__ import annotations
import torch as t
from torchtyping import TensorType
from dictionary_learning import AutoEncoder

class SparseAct():
    """
    A SparseAct is a helper class which represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term.
    A SparseAct may have three fields:
    act : the feature activations in the sparse basis
    res : the SAE error term
    resc : a contracted SAE error term, useful for when we want one number per feature and error (instead of having d_model numbers per error)
    """

    def __init__(
            self, 
            act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None, 
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            resc: TensorType["batch_size", "n_ctx"] = None, # contracted residual
            ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'SparseAct':
        kwargs = {}
        if isinstance(aux, SparseAct):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseAct(**kwargs)
        
    def __mul__(self, other) -> 'SparseAct':
        if isinstance(other, SparseAct):
            # Handle SparseAct * SparseAct
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return SparseAct(**kwargs)

    def __rmul__(self, other) -> 'SparseAct':
        # This will handle float/int * SparseAct by reusing the __mul__ logic
        return self.__mul__(other)
    
    def __matmul__(self, other: SparseAct) -> SparseAct:
        # dot product between two SparseActs, except only the residual is contracted
        return SparseAct(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))
    
    def __add__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) + getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return SparseAct(**kwargs)
    
    def __radd__(self, other: SparseAct) -> SparseAct:
        return self.__add__(other)
    
    def __sub__(self, other: SparseAct) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return SparseAct(**kwargs)
    
    def __truediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseAct(**kwargs)

    def __rtruediv__(self, other) -> SparseAct:
        if isinstance(other, SparseAct):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseAct(**kwargs)

    def __neg__(self) -> SparseAct:
        sparse_result = -self.act
        res_result = -self.res
        return SparseAct(act=sparse_result, res=res_result)
    
    def __invert__(self) -> SparseAct:
            return self._map(lambda x, _: ~x)
    
    def __getitem__(self, index: int):
        return self.act[index]
    
    def __repr__(self):
        if self.res is None:
            return f"SparseAct(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseAct(act={self.act}, res={self.res})"
        else:
            raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")
    
    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseAct(**kwargs)
    
    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseAct(**kwargs)
    
    def nonzero(self):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return SparseAct(**kwargs)
    
    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return SparseAct(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseAct(**kwargs)
    
    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseAct(**kwargs)
    
    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return SparseAct(**kwargs)

    def save(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self
    
    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseAct(act=self.act, res=self.res)
    
    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError("SparseAct has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self

    def __eq__(self, other):
        return self._map(lambda x, y: x == y, other)
    
    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)
    
    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)
    
    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())
    
    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))
    
    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)
    
    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))
    
    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))
    
    def abs(self):
        return self._map(lambda x, _: x.abs())