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
from consistency.utils import T, batch_mul, GaussianFourierProjection,get_timestep_embedding


class Consist(nn.Module):

    """consistency model"""
    embed_dim: int
    hidden_dim: Sequence[int]
    x_dim: int
    t_min:float
    data_std: float
    embedding_type: str
    fourier_scale:float
    dropout_rate: Optional[float] = None
    layer_norm: bool = False
    group_norm: bool = False
    
 
    @nn.compact
    def __call__(self, 
                x_con:jnp.ndarray,
                x:jnp.ndarray, 
                t:jnp.ndarray,
                training: bool=False
                ):
        in_x = batch_mul(x, 1 / jnp.sqrt(t**2 + self.data_std**2))
        cond_t = 0.25 * jnp.log(t)
        
        # timestep/noise_level embedding; only for continuous training
        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            #fourier = partial(FourierFeatures,
                              #output_size=self.embed_dim,
                              #learnable=False)
            #temb=fourier()(cond_t)
            temb = GaussianFourierProjection(
                embedding_size=self.embed_dim, scale=self.fourier_scale
            )(cond_t)
            
        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            temb = get_timestep_embedding(cond_t, self.embed_dim)
        else:
            raise ValueError(f"embedding type {self.embedding_type} unknown.")
            
        time_processor = partial(MLP,
                                 hidden_dims=(self.embed_dim*4, self.embed_dim*4),
                                 activations=nn.swish,
                                 activate_final=True)
        
        time=time_processor()(temb, training=training)
        
        #F(x,t) in consistency model paper
        F_model = partial(MLP,
                          hidden_dims=tuple(list(self.hidden_dim) + [self.x_dim]),
                          activations=nn.swish,
                          layer_norm=self.layer_norm,
                          group_norm=self.group_norm,
                          dropout_rate=self.dropout_rate,
                          activate_final=False)
                              
        input_x = jnp.concatenate([x_con,in_x, time], axis=-1)
        h=F_model()(input_x, training=training)
        
        
        h=batch_mul(
            h, (t - self.t_min) * self.data_std / jnp.sqrt(t**2 + self.data_std**2)
        )
        skip_x = batch_mul(x, self.data_std**2 / ((t - self.t_min) ** 2 + self.data_std**2))
        h = skip_x + h
        
        return h
        
        

                     
                                 