from functools import partial
from typing import Callable, Optional, Sequence, Type,TypeVar
import flax.linen as nn
import jax.numpy as jnp
from networks.initialization import orthogonal_init
import jax


T = TypeVar("T")


def batch_add(a, b):
    return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(
        beta_start, beta_end, timesteps
    )
    return betas


def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas
    
    
    
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
    
    
    
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    embedding_size: int = 256
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,)
        )
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 16
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class OnehotEmbed(nn.Module):
    """
    use learnable null features for un-conditional generation
    """
    # used for conditional embedding
    # the scale of continuous
    # use 0 as the unconditional label
    # set the unconditional feature as zeros = jnp.zeros(self.output_size)
    num_embeddings: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):  # x.shape = (B,)
        x -= 1
        # x takes value [0, N] -> [-1, N-1], treat [-1] as unc. label
        return jax.nn.one_hot(x, self.num_embeddings)  # x.shape = (B, num_embeddings)


class LearnableEmbed(nn.Module):
    """
    use learnable null features for un-conditional generation
    """
    # used for conditional embedding
    # the scale of continuous
    # use 0 as the unconditional label
    # set the unconditional feature as zeros = jnp.zeros(self.output_size)
    output_size: int = 16
    num_embeddings: int = 40
    zero_emtpy_feature: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.zero_emtpy_feature:  # [0, 1, ... , N] -> [-1, 0, 1, ... , N-1], zero = null feature
            x -= 1
        embed = nn.Embed(num_embeddings=self.num_embeddings + int(not self.zero_emtpy_feature),
                         features=self.output_size,
                         embedding_init=orthogonal_init())(x.astype(int))  # (B, output_size)
        if self.zero_emtpy_feature:
            return jnp.where(x[:, jnp.newaxis] < 0, jnp.zeros(self.output_size), embed)
        return embed


class FourierEmbed(nn.Module):
    output_size: int = 16
    rescale: float = 1.

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x[:, jnp.newaxis]
        # assert x.shape[-1] == 1  # (B, 1) x in [0, ..., N] (N+1 classes)
        half_dim = self.output_size // 2
        f = jnp.log(10000) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)  # e.g. [1.   , 0.268, 0.072, 0.019, 0.005, 0.001, 0.   , 0.   ]
        f = x * f / self.rescale
        fourier_embed = jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
        # assign zero embeddings to the null class
        return jnp.where(x <= 0, jnp.zeros(self.output_size), fourier_embed)


