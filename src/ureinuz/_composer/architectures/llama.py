import jax
import jax.numpy as jnp
from dataclasses import replace

from ...models.transformer.common import TransformerCausalLM
from ..utils import registry
from ...models.transformer.llama import LlamaTransformerBlock
from ...configs.transformer import TransformerModelConfig
from ...utils.typing import ShardMode
from ... import nn

class LlamaConfig(TransformerModelConfig):
    pass

class LlamaCausalLM(TransformerCausalLM):
    # Default Megatron-LM style Tensor Parallelism rules
    default_sharding_rules = [
        # (Logical Axis Name, Physical Mesh Axis Name)
        
        # --- Weight Axes ---
        ('vocab', 'tp'),
        ('embed', None),
        ('heads', 'tp'),
        ('kv_heads', 'tp'),
        ('head_dim', None),
        ('mlp', 'tp'),
        
        # --- Activation Axes ---
        ('batch', 'fsdp'),
        ('sequence', None),
    ]

    def __init__(self, config, rngs: nn.Rngs = None, mesh=None, sharding_rules=None):
        if rngs is None:
            rngs = nn.Rngs(0)
            
        dtype = getattr(config, 'dtype', jnp.float32)
        shard_mode = getattr(config, 'shard_mode', ShardMode.AUTO)
        quant = getattr(config, 'quant', None)
        dot_general = getattr(config, 'dot_general', None)
        
        super().__init__(
            config,
            LlamaTransformerBlock,
            embedder=lambda c, s: nn.Embedding(c.vocab_size, c.hidden_size, rngs=s),
            lm_head=lambda c, s: nn.Linear(
                c.hidden_size, c.vocab_size, bias=getattr(c, 'use_bias', False), 
                dtype=dtype, rngs=s, axis_names=('embed', 'vocab'), 
                shard_mode=shard_mode, quant=quant, dot_general=dot_general
            ),
            rngs=rngs
        )
        
        if hasattr(self.embed_tokens, 'embedding'):
            self.embed_tokens.embedding.axis_names = ('vocab', 'embed')
            
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=getattr(config, 'norm_eps', 1e-5), 
            dtype=dtype, shard_mode=shard_mode, axis_names=('embed',)
        )

        if sharding_rules is None:
            sharding_rules = self.default_sharding_rules

        if mesh is not None and shard_mode == ShardMode.EXPLICIT:
            from ...utils.sharding import create_sharding
            self.out_sharding = create_sharding(mesh, ('batch', 'sequence', 'embed'), rules=sharding_rules)
        else:
            self.out_sharding = None

    def __call__(self, x: jax.Array, attention_mask: jax.Array = None, aux = None):
        x = self.embed_tokens(x)
        
        has_cache = aux is not None and getattr(aux, 'key_cache', None) is not None
        if has_cache:
            carry = (x, (aux.key_cache, aux.value_cache), 0)
        else:
            carry = (x, None, 0)
            
        position_idx = getattr(aux, 'position_idx', None)
        is_causal = getattr(aux, 'is_causal', False)
        
        def forward_stack(layer, carry):
            h, full_kv_cache, layer_idx = carry
            current_kv = (full_kv_cache[0][layer_idx], full_kv_cache[1][layer_idx]) if has_cache else None
            
            h, next_cache = layer(
                h, 
                attention_mask=attention_mask, 
                kv_cache=current_kv,
                position_idx=position_idx,
                is_causal=is_causal,
                out_sharding=self.out_sharding
            )
            
            if has_cache:
                new_k_cache = full_kv_cache[0].at[layer_idx].set(next_cache[0])
                new_v_cache = full_kv_cache[1].at[layer_idx].set(next_cache[1])
                full_kv_cache = (new_k_cache, new_v_cache)
                
            return h, full_kv_cache, layer_idx + 1
            
        x, final_kv_cache, _ = self.layers(forward_stack, carry)
        
        x = self.norm(x, out_sharding=self.out_sharding)
        logits = self.lm_head(x, out_sharding=self.out_sharding)
        
        if has_cache:
            aux = replace(aux, 
                key_cache=final_kv_cache[0], 
                value_cache=final_kv_cache[1]
            )
            
        return logits, aux

    @classmethod
    def from_pretrained(cls, path_or_repo, mesh=None, sharding_rules=None, local=False, **kwargs):
        # Load config
        config = LlamaConfig.from_pretrained(path_or_repo, local=local)
        
        # We define how HuggingFace weights map to our components using our new Tuple format
        module_map = [
            ("model.", ""),
            ("input_layernorm", "norm1"),
            ("post_attention_layernorm", "norm2"),
            ("self_attn", "attn"),
            ("embed_tokens.weight", "embed_tokens.embedding"),
        ]
        
        # Call the base class safetensors loader
        # (Note: PretrainedModel.from_pretrained will need to be updated to pass mesh and sharding_rules down!)
        return super().from_pretrained(
            path_or_repo, 
            config=config, 
            module_map=module_map, 
            local=local, 
            mesh=mesh,
            sharding_rules=sharding_rules,
            **kwargs
        )

registry.register(
    'LlamaForCausalLM', 
    LlamaConfig, 
    LlamaCausalLM
)

__all__ = ['LlamaConfig', 'LlamaCausalLM']