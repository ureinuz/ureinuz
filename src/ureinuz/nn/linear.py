import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_uniform, zeros
from .module import Module, Parameter
from ..rng import Rngs

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, seed: Rngs = None, initializer = lecun_uniform()):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if seed is None:
            raise ValueError("A seed must be provided to initialize Linear layer")

        w_key = seed()
        self.weight = Parameter(initializer(w_key, (in_features, out_features), jnp.float32))

        if bias:
            b_key = seed()
            self.bias = Parameter(jnp.zeros((out_features,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        out = jnp.dot(x, self.weight.value)
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}"
