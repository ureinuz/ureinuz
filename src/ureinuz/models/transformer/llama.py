import jax
import jax.numpy as jnp
from dataclasses import replace

from ...configs.transformer import CausalLM, TransformerAuxilialary
from ...layers import RotaryEmbedding, GateMLP, Attention
from ... import nn
from ...utils.typing import ShardMode


class LlamaTransformerBlock(nn.Module):
    def __init__(self, config, seed: nn.Rngs):
        dtype = getattr(config, 'dtype', jnp.float32)
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)
        
        self.norm1 = nn.RMSNorm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, 
            shard_mode=shard_mode, axis_names=('embed',)
        )
        
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            pos_emb=RotaryEmbedding(
                config.head_dim, **(getattr(config, 'posemb_kwargs', {}) or {})
            ),
            bias=config.use_attention_bias,
            dtype=dtype,
            rngs=seed,
            # Fully Granular Logical Axes
            q_axis_names=('embed', 'heads', 'head_dim'),
            k_axis_names=('embed', 'kv_heads', 'head_dim'),
            v_axis_names=('embed', 'kv_heads', 'head_dim'),
            o_axis_names=('heads', 'head_dim', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general
        )
        self.norm2 = nn.RMSNorm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, 
            shard_mode=shard_mode, axis_names=('embed',)
        )
        
        activation_fn = getattr(jax.nn, config.activation_fn)
        self.mlp = GateMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=activation_fn,
            bias=config.use_mlp_bias,
            dtype=dtype,
            rngs=seed,
            # Fully Granular Logical Axes
            gate_axis_names=('embed', 'mlp'),
            up_axis_names=('embed', 'mlp'),
            down_axis_names=('mlp', 'embed'),
            shard_mode=shard_mode,
            quant=quant,
            dot_general=dot_general
        )

    def __call__(
        self, 
        x: jax.Array, 
        attention_mask: jax.Array = None, 
        kv_cache: tuple[jax.Array, jax.Array] = None,
        position_idx: jax.Array = None,
        is_causal: bool = False,
        out_sharding = None
    ):
        attn_out, new_cache = self.attn(
            self.norm1(x), 
            attention_mask=attention_mask, 
            is_causal=is_causal, 
            kv_cache=kv_cache, 
            position_idx=position_idx,
            out_sharding=out_sharding
        )
        
        x = x + attn_out
        x = x + self.mlp(self.norm2(x), out_sharding=out_sharding)
        return x, new_cache



