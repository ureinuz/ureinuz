import jax
import jax.numpy as jnp
from jax.nn.initializers import normal
from .module import Module, Parameter
from ..rng import Rngs

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *, seed: Rngs = None, initializer = normal(0.02)):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        if seed is None:
            raise ValueError("A seed must be provided to initialize Embedding layer")
            
        key = seed()
        self.embedding = Parameter(initializer(key, (num_embeddings, embedding_dim), jnp.float32))
        
    def __call__(self, indices: jax.Array) -> jax.Array:
        return self.embedding[indices]

    def extra_repr(self):
        return f"{self.num_embeddings} → {self.embedding_dim}"
