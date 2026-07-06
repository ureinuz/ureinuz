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
"""A high-performance Deep Learning framework built on JAX"""

class Registry:

    def __init__(self):
        self._registry = {}
        self._classes = set()

    def register(self, key, config, architecture):
        if architecture in self._classes:
            raise ValueError(f'Architecture registration governs many-to-one. found duplicated for {architecture.__name__}')
        
        value = {
            'config': config,
            'class': architecture
        }

        self._classes.add(architecture)
        self._registry[key] = value


    def available(self):
        return list(self._registry.keys())
    
    def available_classes(self):
        return self._classes
    
    def get(self, key):
        return self._registry.get(key)
    
registry = Registry()

__all__ = ['Registry', 'registry']