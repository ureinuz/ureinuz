import jax
import math
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

class SinusoidalPositionalEmbedding(Module):
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        
    def __call__(self, timesteps: jax.Array) -> jax.Array:
        is_scalar = timesteps.ndim == 0
        if is_scalar:
            timesteps = jnp.expand_dims(timesteps, 0)
            
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        # timesteps shape is (B,) or (1,) if it was scalar
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        # If embedding_dim is odd, pad by zero
        if self.embedding_dim % 2 == 1:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
            
        return jnp.squeeze(emb, 0) if is_scalar else emb
