from ...nn.module import Module

import jax
from typing import Callable
from ... import nn, Rngs

class GateMLP(Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        activation: Callable | str = jax.nn.silu,
        bias: bool = False, 
        seed: Rngs = None
    ):
        self.activation = activation if isinstance(activation, Callable) else getattr(jax.nn, activation)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, seed=seed)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, seed=seed)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, seed=seed)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)