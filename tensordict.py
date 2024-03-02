"""
A TensorDict is a dictionary whose keys may be tensors
"""

from dataclasses import dataclass
import torch as t

@dataclass(frozen=True)
class TensorKey:
    x : t.tensor

    def __eq__(self, other):
        return (self.x == other.x).all()

    def __hash__(self):
        return hash(tuple(self.x.tolist()))
    
    def __repr__(self):
        return f"TensorKey({self.x})"
    
class TensorDict:
    _dict : dict

    def __init__(self, d={}):
        self._dict = {TensorKey(k) if isinstance(k, t.Tensor) else k : v for k, v in d.items()}

    def __getitem__(self, key):
        if isinstance(key, t.Tensor):
            key = TensorKey(key)
        return self._dict[key]
    
    def __setitem__(self, key, value):
        if isinstance(key, t.Tensor):
            key = TensorKey(key)
        self._dict[key] = value
    
    def __contains__(self, key):
        if isinstance(key, t.Tensor):
            key = TensorKey(key)
        return key in self._dict
    
    def __delitem__(self, key):
        if isinstance(key, t.Tensor):
            key = TensorKey(key)
        del self._dict[key]
    
    def items(self):
        return (
            (k.x, v) if isinstance(k, TensorKey) else (k, v) for k, v in self._dict.items()
        )
    
    def keys(self):
        return (k.x if isinstance(k, TensorKey) else k for k in self._dict.keys())
    
    def __getattr__(self, name):
        return getattr(self._dict, name)
    
    # magic methods must be defined by hand
    def __len__(self):
        return len(self._dict)
    
    def __repr__(self):
        return f"""TensorDict({dict({
            k : v for k, v in self.items()
        })})"""
    
if __name__ == '__main__':
    x = t.tensor([1,2,3])
    y = t.tensor([1,2,3])
    d = TensorDict({x : 1})
    assert y in d
    d[y] = 2
    assert len(d) == 1
    del d[x]
    assert len(d) == 0
