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
from consistency.consistency import Consist
from consistency.utils import T, batch_mul


@partial(jax.jit,
         static_argnames=('diffuser_sample','tau','loss_norm', 'need_ema'))
def jit_update_f(f: Model,
                 f_tar:Model,
                 diffuser_sample:Callable,
                 batch: collections.defaultdict,
                 rng: PRNGKey,
                 tau: float,
                 T:float,
                 t_min:float,
                 rho:float,
                 num_scales: int,
                 loss_norm: str,
                 need_ema: bool) -> Tuple[PRNGKey, Model,Model, InfoDict]:
                  #num_scales: number of discretization steps
                  
    rng = hk.PRNGSequence(rng)
    
    batch_size = batch["observations"].shape[0]
    x_con=jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
    x=batch["mcreturn"]
    indices = jax.random.randint(next(rng), (batch_size,), 1, num_scales - 1)
    
    t = t_min ** (1 / rho) + (indices) / (num_scales - 1) * (
            T ** (1 / rho) - t_min ** (1 / rho)
        )
    t = t**rho
    t=t[:, jnp.newaxis]
    #print("22222222222222222222222")
    #print(t.shape)
    #print(t_new.shape)

    t2 = t_min ** (1 / rho) + (indices+1) / (num_scales - 1) * (
            T ** (1 / rho) - t_min ** (1 / rho)
        )
    t2 = t2**rho
    t2=t2[:, jnp.newaxis]    
    
    z = jax.random.normal(next(rng), x.shape)
    x_t = x + batch_mul(t2, z)
    dropout_rng = next(rng)

    
    xt_input = jnp.concatenate([x_con,x_t], axis=-1)
    
    #denoiser = diffuser_sample(x_con,x_t, t)
    #score = batch_mul(1 / t**2, denoiser - x)
    score=diffuser_sample(x_con,x_t, t2)
    #print("111111111111111111111111111111111")
    #print((t - t2))
    print("222222222222222222222222222222222")
    print(t)
    print("sssssssssssssssssssssssssssssssss")
    print(score)
    #print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
    #print(z)
    #print("ttttttttttttttttttttttttttttttttt")
    #print(t)
    
    x_t2 = x_t - batch_mul(t - t2, batch_mul(score, t2))
    #print("222222222222222222222222222222222")
    #print(x_t2)
    #xt2_input = jnp.concatenate([x_con,x_t2], axis=-1)

    Ft2= f_tar(x_con,x_t2, t2)
    

    def f_loss_fn(f_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        Ft = f.apply(f_paras,
                     x_con,
                     x_t,
                     t,  # t \in range(1, T+1)
                     rngs={'dropout': dropout_rng},
                     training=True)
                     
        # consistency loss
        diffs = Ft - Ft2
        
        if loss_norm.lower() == "l1":
            losses = jnp.abs(diffs)
            losses = jnp.mean(jnp.squeeze(losses), axis=0)
        elif loss_norm.lower() == "l2":
            losses = diffs**2
            losses = jnp.mean(jnp.squeeze(losses), axis=-1)
        elif loss_norm.lower() == "linf":
            losses = jnp.abs(diffs)
            losses = jnp.max(jnp.squeeze(losses), axis=-1)

        else:
            raise ValueError("Unknown loss norm: {}".format(loss_norm))

        return losses,{'consistency_loss': losses}

    new_f, info = f.apply_gradient(f_loss_fn)
    new_f_tar = ema_update(new_f, f_tar, tau) if need_ema else f_tar

    return next(rng),new_f, new_f_tar,info
    
    



class ConDistill(Agent):

    """discrete consistency distillation"""

    name = "cd"
    model_names = ["f"]
    
    def __init__(self,
                con_x:jnp.array,
                x: jnp.ndarray,
                seed: int,
                diffuser: Agent,
                T: float=80.0,
                f_lr: Union[float, optax.Schedule] = 1e-3,
                lr_decay_steps: int = 2000000,
                tau: float = 0.005,  # ema for critic learning
                update_ema_every: int = 5,
                dropout_rate: Optional[float] = None,
                clip_grad_norm: Optional[float] = None,
                hidden_dims: Sequence[int] = (256, 256, 256),
                layer_norm: bool = False,
                pred_t:int =None,
                data_std: float=0.5,
                t_min:float =0.002,
                time_dim: int = 32,
                rho:float=7.0,
                num_scales: int=1000,
                Qmin:float=0.0,
                Qmax:float=1.0,
                num_samples: int = 50, #number of sampled samples
                loss_norm: str="l1",
                Train:bool=True,
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
    
        if lr_decay_steps is not None:
           f_lr = optax.cosine_decay_schedule(f_lr, lr_decay_steps)
    
        f_def=Consist(hidden_dim= hidden_dims,
                      x_dim=self.x_dim,
                      embedding_type= "fourier",
                      data_std=self.data_std,
                      t_min=self.t_min,
                      time_dim= self.time_dim,
                      dropout_rate=self.dropout_rate,
                      layer_norm=self.layer_norm,
                      #pred_t=pred_t
                      )
        f = Model.create(f_def,
                         inputs=[next(rng), con_x,x, jnp.zeros((1, 1))],  # time
                         optimizer=optax.adam(learning_rate=f_lr))
    
        f_tar = Model.create(f_def,
                             inputs=[next(rng), con_x,x, jnp.zeros((1, 1))])
                     
        # models
        self.f = f
        self.f_tar = f_tar
        self.diffuser = diffuser
    
        self.tau = tau
        self.T=T
        self.t_min=t_min
        self.rho=rho
        self.num_scales=num_scales
    
        self.Qmax=Qmax
        self.Qmin=Qmin
    
        self.num_samples=num_samples
        
        self.loss_norm=loss_norm
        
        # training
        self.rng1 = rng1
        self.rng=rng
        self._n_training_steps = 0
        self.update_ema_every = update_ema_every
        self.Train=Train
    
    def update(self, batch: collections.defaultdict) -> InfoDict:

        info = {}
        # update the consistent model
        need_ema = self._n_training_steps % self.update_ema_every == 0
        self.rng1, self.f, self.f_tar, new_info = jit_update_f(self.f,
                                                              self.f_tar,
                                                              self.diffuser.ncsn,
                                                              batch,
                                                              next(self.rng),
                                                              self.tau,
                                                              self.T,
                                                              self.t_min,
                                                              self.rho,
                                                              self.num_scales,
                                                              self.loss_norm,
                                                              need_ema=need_ema)

        info.update(new_info)

        self._n_training_steps += 1
        print(self._n_training_steps)
        return info 
    
    def onestep_sampler(self,
                        x_con: np.ndarray,
                        rng:PRNGKey,
                        temperature=None,
                        denormalize=False) -> jnp.ndarray:
                        
        if len(x_con.shape) == 1:
            x_con = x_con[jnp.newaxis, :]
        
        x_con = jax.device_put(x_con)
        x_con = x_con.repeat(self.num_samples, axis=0)  # (B*num_samples, dim_obs)
        
        if self.Train:
           self.rng1,key = jax.random.split(self.rng1)
        else:
           self.rng1, key = jax.random.split(rng1)
        
        x_noise = jax.random.normal(key, (x_con.shape[0], self.x_dim)) 
        samples = self.f.apply(self.f.params,
                               x_con,
                               x_noise,
                               #jnp.ones((x_noise.shape[0],))* self.T,
                               jnp.ones((x_noise.shape))* self.T,
                               training=False)
        
        #samples = jnp.clip(samples, -1, 1)
        samples = samples.reshape(-1, self.num_samples, self.x_dim)             
                     
        if denormalize:
           samples = denormalize_value(samples,self.Qmax,self.Qmin)

        return samples
    
