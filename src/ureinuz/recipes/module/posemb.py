import jax
import jax.numpy as jnp
from ... import nn, Rngs

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        # Compute frequencies
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        
        # Duplicate for both halves
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        
        # Cache cos and sin, shaped for broadcasting: [1, seq_len, 1, dim]
        self.cos_cached = jnp.cos(emb)[None, :, None, :]
        self.sin_cached = jnp.sin(emb)[None, :, None, :]

    def __call__(self, q: jax.Array, k: jax.Array, position_idx: jax.Array = None) -> tuple[jax.Array, jax.Array]:
        # q and k are expected to have shape [batch, seq_len, num_heads, head_dim]
        seq_len = q.shape[1]
        
        if position_idx is not None:
            # Use dynamic_slice to extract the correct position frequencies
            cos = jax.lax.dynamic_slice(
                self.cos_cached, 
                (0, position_idx, 0, 0), 
                (1, seq_len, 1, self.cos_cached.shape[-1])
            )
            sin = jax.lax.dynamic_slice(
                self.sin_cached, 
                (0, position_idx, 0, 0), 
                (1, seq_len, 1, self.sin_cached.shape[-1])
            )
        else:
            cos = self.cos_cached[:, :seq_len, ...]
            sin = self.sin_cached[:, :seq_len, ...]
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed
