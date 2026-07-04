import jax
from jax.typing import ArrayLike
from typing import Optional
from jax._src.random.core import PRNGSpecDesc, KeyDTypeLike

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Rngs:
    def __init__(
        self, seed: ArrayLike, *, 
        impl: Optional[PRNGSpecDesc] = None, 
        dtype: Optional[KeyDTypeLike] = None
    ):
        try:
            self._key = jax.random.key(seed, impl=impl, dtype=dtype)
        except TypeError:
            self._key = seed

    def __call__(self):
        self._key, _k = jax.random.split(self._key, 2)
        return _k

    def tree_flatten(self):
        return ((self._key,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj._key = children[0]
        return obj