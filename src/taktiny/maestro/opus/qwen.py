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
"""Qwen architectures"""

from __future__ import annotations

from taktiny.maestro._livret import repertoire
from taktiny.cosettes.transformer._common import TransformerCausalLM
from taktiny.cosettes.transformer.qwen import Qwen2TransformerBlock
from taktiny import nn


class Qwen2CausalLM(TransformerCausalLM):
    def __init__(
        self, config, 
        rngs: nn.Rngs = None, 
        mesh=None, 
        sharding_rules=None
    ):
        super().__init__(
            Qwen2TransformerBlock,
            config=config,
            rngs=rngs,
            mesh=mesh,
            sharding_rules=sharding_rules
        )


repertoire.register('Qwen2ForCausalLM', Qwen2CausalLM)

__all__ = ['Qwen2CausalLM']