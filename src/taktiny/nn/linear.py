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
"""Linear modules"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from taktiny.nn import Rngs

import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_uniform

from taktiny.nn import Module, Parameter
from taktiny.utils.typing import ShardMode, DType


default_linear_initializer = lecun_uniform()

class Linear(Module):
    def __new__(cls, *args, **kwargs):
        if cls is Linear:
            # Determine target quantizer from quant or dtype
            quant = kwargs.get('quant', None)
            dtype = kwargs.get('dtype', None)
            target = str(quant).lower() if quant is not None else (str(dtype).lower() if isinstance(dtype, str) else None)
            
            if target == "fp8":
                instance = super().__new__(LinearFP8)
                instance.__init__(*args, **kwargs)
                return instance
            elif target == "int8":
                instance = super().__new__(LinearINT8)
                instance.__init__(*args, **kwargs)
                return instance
            elif target == "int4":
                instance = super().__new__(LinearINT4)
                instance.__init__(*args, **kwargs)
                return instance
            elif target == "fp4":
                instance = super().__new__(LinearFP4)
                instance.__init__(*args, **kwargs)
                return instance
                
        return super().__new__(cls)

    def __init__(
        self, 
        in_features: int | tuple[int, ...], 
        out_features: int | tuple[int, ...], *,
        bias: bool = True, 
        dtype: DType | str = jnp.float32, 
        rngs: Rngs = None, 
        seed: Rngs = None, 
        initializer = default_linear_initializer,
        quant = None,
        dot_general = None, # Hook for google AQT (aqt_dot_general)
        axis_names: tuple[str | None, ...] | None = None, # Equivalent to kernel_axes in NNX
        shard_mode: ShardMode = ShardMode.AUTO
    ):
        if isinstance(in_features, int):
            in_features = (in_features,)
        else:
            in_features = tuple(in_features)
            
        if isinstance(out_features, int):
            out_features = (out_features,)
        else:
            out_features = tuple(out_features)
            
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.dot_general = dot_general
        self.shard_mode = shard_mode

        if rngs is None and seed is None:
            raise ValueError("A rngs must be provided to initialize Linear layer")
        
        if rngs is None and seed is not None:
            import warnings
            warnings.warn('seed is deprecated. use `rngs` instead')
            rngs = seed

        w_key = rngs()
        weight_shape = in_features + out_features
        self.weight = Parameter(initializer(w_key, weight_shape, dtype, ))
        
        if axis_names is not None:
            assert len(axis_names) == len(weight_shape), f"axis_names length {len(axis_names)} must match weight dims {len(weight_shape)}"
            self.weight.axis_names = axis_names

        if bias:
            b_key = rngs()
            self.bias = Parameter(jnp.zeros(out_features, dtype=dtype))
            if axis_names is not None:
                self.bias.axis_names = axis_names[-len(out_features):]

    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        in_dims = len(self.in_features)
        
        # Contracting dimensions for x are the last `in_dims` dimensions
        x_contracting_dims = tuple(range(x.ndim - in_dims, x.ndim))
        # Contracting dimensions for weight are the first `in_dims` dimensions
        w_contracting_dims = tuple(range(in_dims))
        
        dimension_numbers = ((x_contracting_dims, w_contracting_dims), ((), ()))
        
        if self.dot_general is not None:
            out = self.dot_general(x, self.weight.value, dimension_numbers)
        else:
            import jax
            out = jax.lax.dot_general(x, self.weight.value, dimension_numbers)
            
        if self.shard_mode == ShardMode.EXPLICIT and out_sharding is not None:
            import jax
            out = jax.lax.with_sharding_constraint(out, out_sharding)
            
        if self.use_bias:
            out += self.bias.value
            
        return out

    def extra_repr(self):
        in_str = "x".join(map(str, self.in_features))
        out_str = "x".join(map(str, self.out_features))
        qat_str = " (AQT QAT Enabled)" if self.dot_general is not None else ""
        return f"{in_str} → {out_str}{qat_str}"


class BaseQuantizedLinear(Module):
    def _init_features(self, in_features, out_features, kwargs):
        if isinstance(in_features, int):
            self.in_features = (in_features,)
        else:
            self.in_features = tuple(in_features)
            
        if isinstance(out_features, int):
            self.out_features = (out_features,)
        else:
            self.out_features = tuple(out_features)
            
        import math
        self.in_features_flat = math.prod(self.in_features)
        self.out_features_flat = math.prod(self.out_features)
        
        # Drop unsupported kwargs from Linear signature
        self.use_bias = kwargs.get('bias', False)
        
    def _apply_axis_names(self, kwargs):
        axis_names = kwargs.get('axis_names', None)
        # print(f"DEBUG _apply_axis_names: received kwargs keys: {kwargs.keys()}, axis_names: {axis_names}")
        if axis_names is not None:
            # Flattened representations in quantized layers mean we only have 2 dimensions: in_features_flat and out_features_flat
            # axis_names usually has length = len(in_features) + len(out_features)
            # We will just take the first axis_name of in_features and the first axis_name of out_features
            in_len = len(self.in_features)
            flat_axis_names = (axis_names[0] if in_len > 0 else None, axis_names[in_len] if len(axis_names) > in_len else None)
            
            if hasattr(self, 'weights_q'):
                self.weights_q.axis_names = flat_axis_names
            if hasattr(self, 'scales_q'):
                self.scales_q.axis_names = flat_axis_names
            if hasattr(self, 'scale_of_scales'):
                self.scale_of_scales.axis_names = (None, flat_axis_names[1])
            if hasattr(self, 'bias'):
                self.bias.axis_names = axis_names[-len(self.out_features):]
        
    def _flatten_input(self, x: jax.Array) -> jax.Array:
        in_dims = len(self.in_features)
        return x.reshape(x.shape[:-in_dims] + (self.in_features_flat,))
        
    def _reshape_output(self, out: jax.Array, x: jax.Array) -> jax.Array:
        in_dims = len(self.in_features)
        return out.reshape(x.shape[:-in_dims] + self.out_features)


class LinearFP8(BaseQuantizedLinear):
    def __init__(self, in_features: int | tuple[int, ...], out_features: int | tuple[int, ...], block_size: int = 128, **kwargs):
        self._init_features(in_features, out_features, kwargs)
        self.block_size = block_size
        
        assert self.in_features_flat % block_size == 0
        num_blocks = self.in_features_flat // block_size
        
        self.weights_q = Parameter(jnp.zeros((self.in_features_flat, self.out_features_flat), dtype=jnp.float8_e4m3fn))
        self.scales_q = Parameter(jnp.zeros((num_blocks, self.out_features_flat), dtype=jnp.float8_e4m3fn))
        self.scale_of_scales = Parameter(jnp.zeros((1, self.out_features_flat), dtype=jnp.float32))
        
        if self.use_bias:
            self.bias = Parameter(jnp.zeros(self.out_features, dtype=jnp.float32))

    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        x_flat = self._flatten_input(x)
        
        scales = self.scales_q.value.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        weights = self.weights_q.value.astype(x.dtype) * scales
        
        out = jnp.dot(x_flat, weights)
        out = self._reshape_output(out, x)
        
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, FP8 DoubleQuant(b={self.block_size})"


class LinearINT8(BaseQuantizedLinear):
    def __init__(self, in_features: int | tuple[int, ...], out_features: int | tuple[int, ...], block_size: int = 128, **kwargs):
        self._init_features(in_features, out_features, kwargs)
        self.block_size = block_size
        
        assert self.in_features_flat % block_size == 0
        num_blocks = self.in_features_flat // block_size
        
        self.weights_q = Parameter(jnp.zeros((self.in_features_flat, self.out_features_flat), dtype=jnp.int8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, self.out_features_flat), dtype=jnp.int8))
        self.scale_of_scales = Parameter(jnp.zeros((1, self.out_features_flat), dtype=jnp.float32))
        
        if self.use_bias:
            self.bias = Parameter(jnp.zeros(self.out_features, dtype=jnp.float32))

    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        x_flat = self._flatten_input(x)
        
        scales = self.scales_q.value.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        weights = self.weights_q.value.astype(x.dtype) * scales
        
        out = jnp.dot(x_flat, weights)
        out = self._reshape_output(out, x)
        
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, INT8 DoubleQuant(b={self.block_size})"


class LinearINT4(BaseQuantizedLinear):
    def __init__(self, in_features: int | tuple[int, ...], out_features: int | tuple[int, ...], block_size: int = 128, **kwargs):
        self._init_features(in_features, out_features, kwargs)
        
        if self.in_features_flat % block_size != 0:
            block_size = 64
        assert self.in_features_flat % block_size == 0
        self.block_size = block_size
        
        assert self.out_features_flat % 2 == 0
        num_blocks = self.in_features_flat // block_size
        
        # Pack 2 weights along the out_features dimension into 1 uint8
        self.weights_q = Parameter(jnp.zeros((self.in_features_flat, self.out_features_flat // 2), dtype=jnp.uint8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, self.out_features_flat // 2), dtype=jnp.uint8))
        self.scale_of_scales = Parameter(jnp.zeros((1, self.out_features_flat), dtype=jnp.float32))
        
        if self.use_bias:
            self.bias = Parameter(jnp.zeros(self.out_features, dtype=jnp.float32))
        
        self._apply_axis_names(kwargs)

    def _unpack_int4(self, packed: jax.Array) -> jax.Array:
        low = (packed & 0x0F).astype(jnp.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(jnp.int8) - 8
        unpacked = jnp.stack([low, high], axis=-1)
        return unpacked.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))

    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        x_flat = self._flatten_input(x)
        
        scales_unpacked = self._unpack_int4(self.scales_q.value)
        scales = scales_unpacked.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        
        weights_unpacked = self._unpack_int4(self.weights_q.value)
        weights = weights_unpacked.astype(x.dtype) * scales
        
        out = jnp.dot(x_flat, weights)
        out = self._reshape_output(out, x)
        
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, INT4 DoubleQuant(b={self.block_size})"


class LinearFP4(BaseQuantizedLinear):
    def __init__(self, in_features: int | tuple[int, ...], out_features: int | tuple[int, ...], block_size: int = 128, **kwargs):
        self._init_features(in_features, out_features, kwargs)
        self.block_size = block_size
        
        assert self.in_features_flat % block_size == 0
        assert self.out_features_flat % 2 == 0
        num_blocks = self.in_features_flat // block_size
        
        self.weights_q = Parameter(jnp.zeros((self.in_features_flat, self.out_features_flat // 2), dtype=jnp.uint8))
        self.scales_q = Parameter(jnp.zeros((num_blocks, self.out_features_flat // 2), dtype=jnp.uint8))
        self.scale_of_scales = Parameter(jnp.zeros((1, self.out_features_flat), dtype=jnp.float32))
        
        if self.use_bias:
            self.bias = Parameter(jnp.zeros(self.out_features, dtype=jnp.float32))

    def _unpack_fp4(self, packed: jax.Array) -> jax.Array:
        low = (packed & 0x0F).astype(jnp.int8) - 8
        high = ((packed >> 4) & 0x0F).astype(jnp.int8) - 8
        unpacked = jnp.stack([low, high], axis=-1)
        return unpacked.reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))

    def __call__(self, x: jax.Array, out_sharding=None) -> jax.Array:
        x_flat = self._flatten_input(x)
        
        scales_unpacked = self._unpack_fp4(self.scales_q.value)
        scales = scales_unpacked.astype(x.dtype) * self.scale_of_scales.value.astype(x.dtype)
        scales = jnp.repeat(scales, self.block_size, axis=0)
        
        weights_unpacked = self._unpack_fp4(self.weights_q.value)
        weights = weights_unpacked.astype(x.dtype) * scales
        
        out = jnp.dot(x_flat, weights)
        out = self._reshape_output(out, x)
        
        if self.use_bias:
            out += self.bias.value
        return out

    def extra_repr(self):
        return f"{self.in_features} → {self.out_features}, FP4 DoubleQuant(b={self.block_size})"

__all__ = ['Linear']