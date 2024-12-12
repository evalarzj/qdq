import os
from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
import collections
from networks.model import Model, get_weight_decay_mask
from networks.mlp import MLP
from networks.updates import ema_update
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent
from utils import sample_n_k, sampleqdata, denormalize_value

import haiku as hk
from scorematch.scoremodel import ScoreNet
from scorematch.utils import T, batch_mul
from scorematch import sde_lib
#from score.sde_lib import SDE


@partial(jax.jit,
         static_argnames=('sde','train'))
def jit_update_score(score: Model,
                     sde:sde_lib.SDE,
                     batch: collections.defaultdict,
                     rng: PRNGKey,
                     train: bool,
                     likelihood_weighting: bool=False,
                     #ssm:bool=False,
                     eps: float=1e-5) -> Tuple[PRNGKey, Model,InfoDict]:
                 
                  
    rng = hk.PRNGSequence(rng)
    dropout_rng = next(rng)
    
    batch_size = batch["observations"].shape[0]
    x_con=jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
    x=batch["mcreturn"]
    
    if isinstance(sde, sde_lib.KVESDE):
        t = jax.random.normal(next(rng), (x.shape[0],)) * 1.2 - 1.2
        t = jnp.exp(t)
    else:
        t = jax.random.uniform(next(rng), (x.shape[0],), minval=eps, maxval=sde.T)
    
    z = jax.random.normal(next(rng), x.shape)
    
    mean, std = sde.marginal_prob(x, t)
    perturbed_x = mean + batch_mul(std, z)
    
   
    xt_input = jnp.concatenate([x_con,perturbed_x], axis=-1)
    in_x = batch_mul(perturbed_x, 1 / jnp.sqrt(t**2 + sde.data_std**2))
    cond_t = 0.25 * jnp.log(t)

    def score_loss_fn(score_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
    
        denoiser = score.apply(score_paras,
                               x_con,
                               in_x,
                               cond_t,
                               rngs={'dropout': dropout_rng},
                               training=train)
                                      
        denoiser = batch_mul(
                denoiser, t * sde.data_std / jnp.sqrt(t**2 + sde.data_std**2)
            )
        
        skip_x = batch_mul(perturbed_x, sde.data_std**2 / (t**2 + sde.data_std**2))
        denoiser = skip_x + denoiser
        
        scoret = batch_mul(denoiser - perturbed_x, 1 / t**2)
        
        losses = jnp.square(batch_mul(scoret, std) + z)
        losses = batch_mul(
                losses, (std**2 + sde.data_std**2) / sde.data_std**2
            )
        
        #losses = jnp.sum(losses.reshape((losses.shape[0], -1)), axis=-1)
        losses= jnp.mean(jnp.squeeze(losses),axis=-1)
        
        return losses,{'score_matching_loss': losses}

    new_score, info = score.apply_gradient(score_loss_fn)

    return next(rng),new_score, info
    
    



class ScoreLearner(Agent):

    """Noice conditional score matching"""

    name = "sm"
    model_names = ["score"]
    
    def __init__(self,
                con_x:jnp.array,
                x: jnp.ndarray,
                mc_return_ori: jnp.ndarray,
                seed: int,
                T: float=80.0,
                score_lr: Union[float, optax.Schedule] = 1e-3,
                lr_decay_steps: int = 2000000,
                dropout_rate: Optional[float] = None,
                hidden_dims: Sequence[int] = (256, 256, 256),
                layer_norm: bool = False,
                group_norm: bool = True,
                data_std: float=0.5,
                t_min:float =0.002,
                time_dim: int = 32,
                rho:float=7.0,
                num_scales: int=18,
                fourier_scale: int=16,
                sde_name :str="KVESDE",
                Qmin:float=0.0,
                Qmax:float=1.0,
                num_samples: int = 50, #number of sampled samples
                Train:bool=True,
                train_model: bool=True,
                likelihood_weighting: bool=False,
                eps: float=1e-5,
                **kwargs,
                 ):   
                 
        rng1 = jax.random.PRNGKey(seed)
        rng = hk.PRNGSequence(rng1)
    
        self.x_dim = x.shape[-1]
        self.data_std=data_std
        self.t_min=t_min
        self.time_dim=time_dim
        self.dropout_rate=dropout_rate
        self.layer_norm=layer_norm
        self.group_norm=group_norm
        self.fourier_scale=fourier_scale
        self.likelihood_weighting=likelihood_weighting
    
        if lr_decay_steps is not None:
           score_lr = optax.cosine_decay_schedule(score_lr, lr_decay_steps)
           
    
        score_def=ScoreNet(embed_dim=self.time_dim,
                           hidden_dim= hidden_dims,
                           x_dim=self.x_dim,
                           embedding_type= "fourier",
                           data_std=self.data_std,
                           fourier_scale=self.fourier_scale,
                           dropout_rate=self.dropout_rate,
                           layer_norm=self.layer_norm,
                           group_norm=self.group_norm,
                      )
        score = Model.create(score_def,
                         inputs=[next(rng), con_x,x, jnp.zeros((1))],
                         optimizer=optax.adam(learning_rate=score_lr))
    
                     
        # models
        self.score = score
    
        self.T=T
        self.t_min=t_min
        self.rho=rho
        self.num_scales=num_scales
    
        self.Qmax=Qmax
        self.Qmin=Qmin
    
        self.num_samples=num_samples
        self.eps=eps
        
        
        # training
        self.rng1 = rng1
        self.rng=rng
        self._n_training_steps = 0
        self.Train=Train
        self.train_model=train_model
        
        #maximium and minimium for the original mc_return(used for scale the Q value)
        if self.Train:
           self.Qmax=jnp.max(mc_return_ori, axis=0)
           self.Qmin=jnp.min(mc_return_ori, axis=0)
        else:
           self.Qmax=Qmax
           self.Qmin=Qmin
        
        self.sde_name=sde_name
        
        # Setup SDEs
        sde = sde_lib.get_sde(self.sde_name,
                              t_min = self.t_min,
                              t_max = self.T,
                              num_scales = self.num_scales,
                              rho = self.rho,
                              data_std = self.data_std)
                              
        self.sde = sde
    
    def denoiser_fn(self, x_con,x, t,train=False,return_state=False, rng=None):
        assert isinstance(
        self.sde, sde_lib.KVESDE
        ), "Only KVE SDE is supported for building the denoiser"
        in_x = batch_mul(x, 1 / jnp.sqrt(t**2 + self.sde.data_std**2))
        cond_t = 0.25 * jnp.log(t)
        denoiser = self.score.apply(self.score.params,x_con,in_x, cond_t, rngs=rng,training=train)
        denoiser = batch_mul(
                denoiser, t * self.sde.data_std / jnp.sqrt(t**2 + self.sde.data_std**2)
            )
        skip_x = batch_mul(x, self.sde.data_std**2 / (t**2 + self.sde.data_std**2))
        denoiser = skip_x + denoiser

        if return_state:
           return denoiser, state
        else:
           return denoiser

        return denoiser
    
    def update(self, batch: collections.defaultdict) -> InfoDict:

        info = {}
        # update the consistent model
        self.rng1, self.score, new_info = jit_update_score(self.score,
                                                           self.sde,
                                                           batch,
                                                           next(self.rng),
                                                           self.train_model,
                                                           self.likelihood_weighting,
                                                           self.eps)

        info.update(new_info)

        self._n_training_steps += 1
        print(self._n_training_steps)
        return info 
    
    def get_heun_sampler(self, x_con, shape, rng ,denoise=True,denormalize=False):
        x_con = jax.device_put(x_con)
        x_con = x_con.repeat(self.num_samples, axis=0)  # (B*num_samples, dim_obs)
        
        def heun_sampler(x_con, shape, rng):
            rng = hk.PRNGSequence(rng)
            x = self.sde.prior_sampling(next(rng), shape)
            
            timesteps = (
                self.sde.t_max ** (1 / self.sde.rho)
                + jnp.arange(self.sde.N)
                / (self.sde.N - 1)
                * (self.sde.t_min ** (1 / self.sde.rho) - self.sde.t_max ** (1 / self.sde.rho))
            ) ** self.sde.rho
            timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])

            def loop_body(i, val):
                x = val
                t = timesteps[i]
                vec_t = jnp.ones((shape[0],)) * t
                denoiser = self.denoiser_fn(x_con,x, vec_t)
                d = 1 / t * x - 1 / t * denoiser
                next_t = timesteps[i + 1]
                samples = x + (next_t - t) * d

                vec_next_t = jnp.ones((shape[0],)) * next_t
                denoiser = self.denoiser_fn(x_con,samples, vec_next_t)
                next_d = 1 / next_t * samples - 1 / next_t * denoiser
                samples = x + (next_t - t) / 2 * (d + next_d)

                return samples
            x = jax.lax.fori_loop(0, self.sde.N - 1, loop_body, x)
            if denoise:
                t = timesteps[self.sde.N - 1]
                vec_t = jnp.ones((shape[0],)) * t
                denoiser = self.denoiser_fn(x_con,x, vec_t)
                d = 1 / t * x - 1 / t * denoiser
                next_t = timesteps[self.sde.N]
                samples = x + (next_t - t) * d
            else:
                samples = x
            
            samples = samples.reshape(-1, self.num_samples, self.x_dim)
            
            if denormalize:
               samples = denormalize_value(samples,self.Qmax,self.Qmin)
                
            return samples, self.sde.N
        

        return heun_sampler(x_con, shape, rng)
    
    def get_euler_sampler(self, x_con,shape,rng, denoise=True,denormalize=False):
    
        x_con = jax.device_put(x_con)
        x_con = x_con.repeat(self.num_samples, axis=0)  # (B*num_samples, dim_obs)
        
        def euler_sampler(x_con,shape,rng):
            rng = hk.PRNGSequence(rng)
            x = self.sde.prior_sampling(next(rng), shape)
            #x = jnp.concatenate([x_con,x], axis=-1)
            timesteps = (
                self.sde.t_max ** (1 / self.sde.rho)
                + jnp.arange(self.sde.N)
                / (self.sde.N - 1)
                * (self.sde.t_min ** (1 / self.sde.rho) - self.sde.t_max ** (1 / self.sde.rho))
            ) ** self.sde.rho
            timesteps = jnp.concatenate([timesteps, jnp.array([0.0])])

            def loop_body(i, val):
                x = val
                t = timesteps[i]
                vec_t = jnp.ones((shape[0],)) * t
                denoiser = self.denoiser_fn(x_con,x, vec_t)
                d = 1 / t * x - 1 / t * denoiser
                next_t = timesteps[i + 1]
                samples = x + (next_t - t) * d
                return samples

            x = jax.lax.fori_loop(0, self.sde.N - 1, loop_body, x)
            if denoise:
                t = timesteps[self.sde.N - 1]
                vec_t = jnp.ones((shape[0],)) * t
                denoiser = self.denoiser_fn(x_con,x, vec_t)
                d = 1 / t * x - 1 / t * denoiser
                next_t = timesteps[self.sde.N]
                samples = x + (next_t - t) * d
            else:
                samples = x
                
            samples = samples.reshape(-1, self.num_samples, self.x_dim)
            
            if denormalize:
               samples = denormalize_value(samples,self.Qmax,self.Qmin)
               
            return samples, self.sde.N

        return euler_sampler(x_con, shape, rng)

    
