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
         static_argnames=('diffuser_sample','tau','loss_norm', 'need_ema','train','solver','dsm_target'))
def jit_update_f(f: Model,
                 f_tar:Model,
                 diffuser_sample:Callable,
                 batch: collections.defaultdict,
                 rng: PRNGKey,
                 train: bool,
                 tau: float,
                 T:float,
                 t_min:float,
                 rho:float,
                 num_scales: int,
                 loss_norm: str,
                 need_ema: bool,
                 solver: str,
                 dsm_target: bool) -> Tuple[PRNGKey, Model,Model, InfoDict]:
                  #num_scales: number of discretization steps
                  
    rng = hk.PRNGSequence(rng)
    
    batch_size = batch["observations"].shape[0]
    x_con=jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)
    x=batch["mcreturn"]
    
    dropout_rng = next(rng)
    
    def heun_solver(x_con,samples, t, next_t, x0):
        x = samples
        if dsm_target:
            denoiser = x0
        else:
            denoiser = diffuser_sample(x_con,x, t)

        d = batch_mul(1 / t, x - denoiser)

        samples = x + batch_mul(next_t - t, d)
        if dsm_target:
            denoiser = x0
        else:
            denoiser = diffuser_sample(x_con,samples, next_t)
        next_d = batch_mul(1 / next_t, samples - denoiser)
        samples = x + batch_mul((next_t - t) / 2, d + next_d)

        return samples

    def euler_solver(x_con,samples, t, next_t, x0):
        x = samples
        if dsm_target:
            denoiser = x0
        else:
            denoiser = diffuser_sample(x_con,x, t)
        score = batch_mul(1 / t**2, denoiser - x)
        samples = x + batch_mul(next_t - t, -batch_mul(score, t))

        return samples

    if solver.lower() == "heun":
        ode_solver = heun_solver
    elif solver.lower() == "euler":
        ode_solver = euler_solver
        
    

    def f_loss_fn(f_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
    
        indices = jax.random.randint(next(rng), (batch_size,), 1, num_scales - 1)
    
        t = T ** (1 / rho) + (indices) / (num_scales - 1) * (
            t_min ** (1 / rho) - T ** (1 / rho)
        )
        t = t**rho
        #t=t[:, jnp.newaxis]

        t2 = T ** (1 / rho) + (indices+1) / (num_scales - 1) * (
            t_min ** (1 / rho) - T ** (1 / rho)
        )
        t2 = t2**rho
        #t2=t2[:, jnp.newaxis]    
    
        z = jax.random.normal(next(rng), x.shape)
        x_t = x + batch_mul(t, z)
        
        
        Ft = f.apply(f_paras,
                     x_con,
                     x_t,
                     t,  # t \in range(1, T+1)
                     rngs={'dropout': dropout_rng},
                     training=train)
    
        x_t2=ode_solver(x_con,x_t, t,t2,x)

        Ft2= f_tar(x_con,x_t2, t2)
    
        
                     
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
                group_norm: bool = True,
                pred_t:int =None,
                data_std: float=0.5,
                t_min:float =0.002,
                time_dim: int = 32,
                rho:float=7.0,
                num_scales: int=18,
                fourier_scale: int=16,
                Qmin:float=0.0,
                Qmax:float=1.0,
                num_samples: int = 50, #number of sampled samples
                loss_norm: str="l1",
                Train: bool=True,
                solver: str="heun",
                dsm_target:bool=False,
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
        self.Train=Train
        self.solver=solver
        self.dsm_target=dsm_target
    
        if lr_decay_steps is not None:
           f_lr = optax.cosine_decay_schedule(f_lr, lr_decay_steps)
    
        f_def=Consist(embed_dim=self.time_dim,
                      hidden_dim= hidden_dims,
                      x_dim=self.x_dim,
                      t_min=self.t_min,
                      data_std=self.data_std,
                      embedding_type= "fourier",
                      fourier_scale=self.fourier_scale,
                      dropout_rate=self.dropout_rate,
                      layer_norm=self.layer_norm,
                      group_norm=self.group_norm,
                      )
        f = Model.create(f_def,
                         inputs=[next(rng), con_x,x, jnp.zeros((1))],  # time
                         optimizer=optax.radam(learning_rate=f_lr))
    
        f_tar = Model.create(f_def,
                             inputs=[next(rng), con_x,x, jnp.zeros((1))])
                     
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

    
    def update(self, batch: collections.defaultdict) -> InfoDict:

        info = {}
        # update the consistent model
        need_ema = self._n_training_steps % self.update_ema_every == 0
        self.rng1, self.f, self.f_tar, new_info = jit_update_f(self.f,
                                                               self.f_tar,
                                                               self.diffuser.denoiser_fn,
                                                               batch,
                                                               next(self.rng),
                                                               self.Train,
                                                               self.tau,
                                                               self.T,
                                                               self.t_min,
                                                               self.rho,
                                                               self.num_scales,
                                                               self.loss_norm,
                                                               need_ema=need_ema,
                                                               solver=self.solver,
                                                               dsm_target=self.dsm_target)

        info.update(new_info)

        self._n_training_steps += 1
        print(self._n_training_steps)
        return info 
    
    def onestep_sampler(self,
                        x_con: np.ndarray,
                        rng:PRNGKey,
                        denormalize=False) -> jnp.ndarray:
                        
        if len(x_con.shape) == 1:
            x_con = x_con[jnp.newaxis, :]
        
        x_con = jax.device_put(x_con)
        x_con = x_con.repeat(self.num_samples, axis=0)  # (B*num_samples, dim_obs)
        
        if self.Train:
           self.rng1,key = jax.random.split(self.rng1)
        else:
           self.rng1, key = jax.random.split(rng1)
        
        x_noise = jax.random.normal(key, (x_con.shape[0], self.x_dim))*self.T
        
        
        samples = self.f.apply(self.f.params,
                               x_con,
                               x_noise,
                               jnp.ones((x_noise.shape[0],))* self.T,
                               #jnp.ones((x_noise.shape))* self.T,
                               training=False)
        
        #samples = jnp.clip(samples, -1, 1)
        samples = samples.reshape(-1, self.num_samples, self.x_dim)             
                     
        if denormalize:
           samples = denormalize_value(samples,self.Qmax,self.Qmin)

        return samples
    
