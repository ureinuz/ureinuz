import jax
import jax.numpy as jnp
from .module import Module, Parameter

class LayerNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(jnp.ones((hidden_size,), dtype=jnp.float32))
        self.bias = Parameter(jnp.zeros((hidden_size,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.eps)
        return x_norm * self.weight.value + self.bias.value

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"

class RMSNorm(Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_norm = x * jax.lax.rsqrt(var + self.eps)
        return x_norm * self.weight.value

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"
