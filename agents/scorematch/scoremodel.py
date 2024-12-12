from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
from networks.model import Model
from networks.scoremlp import MLP
from networks.types import Params, InfoDict, PRNGKey, Batch
import flax.linen as nn
import jax.numpy as jnp
import jax

import flax
import haiku as hk
from scorematch.utils import T, batch_mul, GaussianFourierProjection,get_timestep_embedding


class ScoreNet(nn.Module):

    """score mathching model"""
    embed_dim: int
    hidden_dim: Sequence[int]
    x_dim: int
    embedding_type: str
    data_std: float
    fourier_scale:float
    dropout_rate: Optional[float] = None
    layer_norm: bool = False
    group_norm: bool = False
    #marginal_prob_std: Any
 
 
    @nn.compact
    def __call__(self, 
                 x_con:jnp.ndarray,
                 x:jnp.ndarray, 
                 t:jnp.ndarray,
                 training: bool=False
                ):

        
        # timestep/noise_level embedding; only for continuous training
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            temb = GaussianFourierProjection(
                embedding_size=self.embed_dim, scale=self.fourier_scale
            )(t)
        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            temb = get_timestep_embedding(t, self.time_dim)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")
            
        time_processor = partial(MLP,
                                 hidden_dims=(self.embed_dim*4, self.embed_dim*4),
                                 activations=nn.swish,
                                 activate_final=True)
        
        time=time_processor()(temb, training=training)
        
        
        score = partial(MLP,
                        hidden_dims=tuple(list(self.hidden_dim) + [self.x_dim]),
                        activations=nn.swish,
                        layer_norm=self.layer_norm,
                        group_norm=self.group_norm,
                        dropout_rate=self.dropout_rate,
                        activate_final=False)
                              
        input_x = jnp.concatenate([x_con,x, time], axis=-1)
        h=score()(input_x, training=training)
        
        # Normalize output
        #mean, std =self.marginal_prob_std(x,t)
        #h = h / std
        
        return h
        
        

                     
                                 