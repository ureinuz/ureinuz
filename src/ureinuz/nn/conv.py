import jax
import jax.numpy as jnp
from .module import Module, Parameter
from .. import Rngs
import math

class Conv2d(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int | tuple[int, int], 
        stride: int | tuple[int, int] = 1,
        padding: str | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = "SAME",
        groups: int = 1,
        use_bias: bool = True,
        seed: Rngs = None
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
            
        in_channels_per_group = in_channels // groups
        
        # LeCun uniform initialization
        k = math.sqrt(1.0 / (in_channels_per_group * kernel_size[0] * kernel_size[1]))
        
        if seed is not None:
            w_key = seed()
            b_key = seed()
            # Shape: (H, W, I, O)
            w_shape = (*kernel_size, in_channels_per_group, out_channels)
            w_init = jax.random.uniform(w_key, w_shape, minval=-k, maxval=k)
            
            if use_bias:
                b_init = jax.random.uniform(b_key, (out_channels,), minval=-k, maxval=k)
            else:
                b_init = None
        else:
            w_shape = (*kernel_size, in_channels_per_group, out_channels)
            w_init = jnp.zeros(w_shape)
            b_init = jnp.zeros((out_channels,)) if use_bias else None
            
        self.weight = Parameter(w_init)
        if use_bias:
            self.bias = Parameter(b_init)
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        # x shape: (N, H, W, C)
        out = jax.lax.conv_general_dilated(
            lhs=x, 
            rhs=self.weight.value,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.groups
        )
        
        if self.use_bias:
            # Bias is (C,), broadcasting automatically handles NHWC
            out = out + self.bias.value
            
        return out
