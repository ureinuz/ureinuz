from ...models.transformer.common import TransformerCausalLM
from ..utils import registry
# Provide how to
# - compose
# - shard 
# - init from pretrained weight

class Qwen3Config:
    ...


class Qwen3CausalLM(TransformerCausalLM):
    def __init__(self):
        ...


registry.register(
    'Qwen3ForCausalLM', 
    Qwen3Config, 
    Qwen3CausalLM
)

__all__ = ['Qwen3Config', 'Qwen3CausalLM']