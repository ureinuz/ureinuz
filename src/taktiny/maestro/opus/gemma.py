# Copyright 2026 Shinapri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma architectures"""

from __future__ import annotations

import jax.numpy as jnp

from taktiny.maestro._livret import repertoire
from taktiny.cosettes.transformer._common import TransformerCausalLM
from taktiny.cosettes.transformer.gemma import GemmaTextScaledWordEmbedding, GemmaRMSNorm, GemmaTransformerBlock
from taktiny import nn


class GemmaCausalLM(TransformerCausalLM):
    def __init__(
        self, 
        config, 
        rngs: nn.Rngs = None, 
        mesh=None, 
        sharding_rules=None
    ):
        super().__init__(
            GemmaTransformerBlock,
            GemmaTextScaledWordEmbedding,
            config=config,
            rngs=rngs,
            mesh=mesh,
            sharding_rules=sharding_rules
        )
            
        self.norm = GemmaRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps, 
            dtype=jnp.float32, 
            shard_mode=self.shard_mode, 
            axis_names=('embed',)
        )

    @classmethod
    def from_pretrained(cls, path_or_repo, mesh=None, sharding_rules=None, local=False, **kwargs):
        # Load config
        from taktiny.maestro._config import ModelConfig
        config = ModelConfig.load_config(path_or_repo, local=local)
        
        # Gemma models always tie word embeddings, but the HF config.json might not explicitly have this field
        config.tie_word_embeddings = True
        
        return super().from_pretrained(
            path_or_repo, 
            mesh=mesh, 
            sharding_rules=sharding_rules, 
            local=local, 
            config=config
        )
    

repertoire.register('GemmaForCausalLM', GemmaCausalLM)

__all__ = ['GemmaCausalLM']