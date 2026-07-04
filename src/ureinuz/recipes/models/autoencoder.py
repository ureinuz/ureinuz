import jax
import jax.numpy as jnp
from ... import nn, Rngs
from .config.cnn import CNNModelConfig

class Encoder(nn.Module):
    def __init__(self, config: CNNModelConfig, seed: Rngs):
        self.layers = nn.List()
        in_channels = config.in_channels
        
        for dim in config.dims:
            # Downsample using Conv2d with stride=2
            self.layers.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding="SAME", seed=seed),
                    nn.LayerNorm(dim)
                )
            )
            in_channels = dim
            
        self.latent_proj = nn.Conv2d(in_channels, config.latent_dim, kernel_size=1, seed=seed)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.gelu(x)
        return self.latent_proj(x)

class Decoder(nn.Module):
    def __init__(self, config: CNNModelConfig, seed: Rngs):
        self.layers = nn.List()
        in_channels = config.latent_dim
        
        # Reverse the hidden dims
        reversed_dims = list(reversed(config.dims))
        
        for dim in reversed_dims:
            # Upsample back using Upsample2d + Conv2d
            self.layers.layers.append(
                nn.Sequential(
                    nn.Upsample2d(scale_factor=2),
                    nn.Conv2d(in_channels, dim, kernel_size=3, padding="SAME", seed=seed),
                    nn.LayerNorm(dim)
                )
            )
            in_channels = dim
            
        self.final_proj = nn.Conv2d(in_channels, config.in_channels, kernel_size=3, padding="SAME", seed=seed)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.gelu(x)
        return self.final_proj(x)

class Autoencoder(nn.Module):
    def __init__(self, config: CNNModelConfig, seed: Rngs):
        self.encoder = Encoder(config, seed=seed)
        self.decoder = Decoder(config, seed=seed)
        
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
