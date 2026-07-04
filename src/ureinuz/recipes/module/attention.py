import jax
import jax.numpy as jnp
from ... import nn, Rngs

from .posemb import RotaryEmbedding

class Attention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        head_dim: int, 
        num_kv_heads: int = None,
        pos_emb: nn.Module = None,
        bias: bool = False, 
        seed: Rngs = None
    ):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else self.num_heads
        
        # For Grouped Query Attention (GQA)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.pos_emb = pos_emb
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias, seed=seed)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias, seed=seed)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias, seed=seed)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias, seed=seed)
        
    def __call__(
        self, 
        x: jax.Array, 
        attention_mask: jax.Array = None, 
        is_causal: bool = False,
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None
    ) -> jax.Array | tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        # x shape: [Batch, SeqLen, HiddenSize]
        B, L, _ = x.shape
        
        # Project and reshape to [B, L, Heads, HeadDim]
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        # Apply Positional Embeddings (if provided)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, position_idx)
        
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Update cache at position_idx using dynamic_update_slice
            k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, position_idx, 0, 0))
            v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, position_idx, 0, 0))
            
            # Use the full cache for attention
            k = k_cache
            v = v_cache
            
        # JAX's dot_product_attention natively handles GQA (num_kv_heads < num_heads)!
        # It expects shape [Batch, SeqLen, Heads, HeadDim] for query, key, and value.
        # If mask is boolean, it filters out logits. If additive, pass to `bias`.
        out = jax.nn.dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            mask=attention_mask,
            is_causal=is_causal
        )
        
        # Flatten heads: [B, L, Heads * HeadDim]
        out = out.reshape(B, L, -1)
        
        # Output projection
        out = self.o_proj(out)
        
        if kv_cache is not None:
            return out, (k_cache, v_cache)
        
        return out, None