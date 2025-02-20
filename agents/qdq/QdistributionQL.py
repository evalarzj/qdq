#  Q distribution based Q learning

import os
from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
from networks.model import Model, get_weight_decay_mask
from networks.policies import NormalTanhPolicy#, sample_actions
from networks.policies import MSEPolicy
from networks.critics import MultiHeadQ
from networks.updates import ema_update
#from diffusions.diffusion import DDPM, ddpm_sampler, ddim_sampler
#from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent

"""Implementations of algorithms for continuous control."""


def mish(x):
    return x * jnp.tanh(nn.softplus(x))
    

@partial(jax.jit, static_argnames=('network','deterministic'))
def _sample_actions(rng: PRNGKey,
                    network: nn.Module,
                    params: Params,
                    observations: np.ndarray,
                    temperature: float,
                    deterministic: bool = False) -> Tuple[PRNGKey, jnp.ndarray]:
    dist,sample_action = network.apply(params, observations,temperature)
    rng, key = jax.random.split(rng)
    
    #return rng, sample_action
    if deterministic:
       return rng, sample_action
    else:
       return rng, dist.sample(seed=key)



@partial(jax.jit,
         static_argnames=('tau','need_ema','gamma','env','awr'))
def jit_update_actor(actor: Model,
                     actor_tar: Model,
                     critic: Model,
                     batch: Batch,
                     rng: PRNGKey,
                     env: bool,
                     awr: bool,
                     tau: float,
                     gamma:float,
                     need_ema: bool) -> Tuple[PRNGKey, Model, Model, InfoDict]:


    rng, tr_key,dropoutkey,samplekey= jax.random.split(rng, 4)
    batch_size = batch.observations.shape[0]
    

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        
        # Q-distribution based Q learning
        dist,actions = actor.apply(actor_params,
                                   batch.observations,
                                   training=True,
                                   rngs={'dropout': dropoutkey})
                              
        #actions=dist.sample(seed=samplekey)
        log_probs = dist.log_prob(batch.actions)
        q3, q4 = critic(batch.observations, actions)
        qq = jnp.minimum(q3, q4)
        
        if env and awr:
            q1, q2 = critic(batch.observations, batch.actions)
            q = jnp.minimum(q1, q2)
            rep_obs = batch.observations.repeat(30, axis=0)
            dist,_ = actor_tar(rep_obs)
            rep_actions=dist.sample(seed=tr_key)
            rep_Q = critic(rep_obs, rep_actions).min(axis=0).reshape(batch_size, -1)
            rep_v = rep_Q.mean(axis=-1)
            exp_a = jnp.exp((q-rep_v) * 6)
            exp_a = jnp.minimum(exp_a, 100.0)  # truncate the weights...
            actor_loss = -(exp_a * log_probs).mean()
        else:
            actor_loss = -(gamma*log_probs).mean()-qq.mean()

        return actor_loss, {'actor_loss': actor_loss,'Qguidance':qq.mean(),'logprob':log_probs}#,'distance':exp_a,

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    new_actor_tar = ema_update(new_actor, actor_tar, tau) if need_ema else actor_tar
    return rng, new_actor, new_actor_tar, info


@partial(jax.jit,
         static_argnames=('discount', 'tau', 'critic_sample','need_ema','random_policy',"env","awr"))
def _jit_update_critic(rng: PRNGKey,
                       critic: Model,
                       critic_tar: Model,
                       actor_tar: Model,
                       discount: float,
                       env: bool,
                       awr: bool,
                       tau: float,
                       critic_sample: Callable,
                       batch: Batch,
                       alpha : float,
                       beta: float,
                       need_ema: bool,
                       random_policy:bool,
                       target_policy_noise:float=0.2,
                       target_policy_noise_clip:float=0.5
                       ):
    rng, key,samplekey = jax.random.split(rng,3)
    batch_size = batch.observations.shape[0]

    # create next step action
    dist,next_actions = actor_tar(batch.next_observations)
    
    
    if random_policy:
       next_actions=dist.sample(seed=key)
    
    

    noise = jax.random.normal(key,next_actions.shape) * target_policy_noise
    noise = jnp.clip(
            noise,
            -target_policy_noise_clip,
            target_policy_noise_clip)
    noisy_next_actions = next_actions + noise
    noisy_next_actions = jnp.clip(noisy_next_actions,-1,1)
    
    #estimate the uncertainty of next step Q value
    obs_action=jnp.concatenate([batch.next_observations, noisy_next_actions], axis=-1)
    #sampling_shape = (batch_size*num_samples,1)
    #q,n=critic_sample(obs_action,shape=sampling_shape,rng=samplekey,denormalize=True)
    q=critic_sample(obs_action,rng=samplekey,denormalize=True)
    
    qstd_o=jnp.std(q,axis=1)
    qstd_o = jnp.squeeze(qstd_o)
    
    
    qstd_sd=(jnp.max(qstd_o)-qstd_o)/(jnp.max(qstd_o)-jnp.min(qstd_o))
    qstd_m=qstd_o-jnp.quantile(qstd_o,beta)
    
    if env and awr:
        qstd = (qstd_m> 0) * (1+0.1*(1-1/qstd_o)) + (qstd_m<= 0) * 1
    else:
        if env and not awr:
           qstd = (qstd_m> 0) * (1/qstd_o) + (qstd_m<= 0) * 1
        else:
           qstd = (qstd_m> 0) * (1/qstd_o) + (qstd_m<= 0) * beta
   
    
    
    
    # find the Q-target and weigth with the standard deviation
    tar_q1, tar_q2 = critic_tar(batch.next_observations, noisy_next_actions)
    
    tar_q = jnp.minimum(tar_q1, tar_q2)
    target_qh = batch.rewards + discount * batch.masks * tar_q
    
    tar_q2 = jnp.multiply(tar_q,qstd)
    target_ql = batch.rewards + discount * batch.masks * tar_q2

    

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply(critic_params, batch.observations,
                              batch.actions)
        
        critic_lossh = ((q1 - target_qh) ** 2 + (q2 - target_qh) ** 2).mean()
        critic_lossl = ((q1 - target_ql) ** 2 + (q2 - target_ql) ** 2).mean()

        critic_loss=alpha*critic_lossh+(1-alpha)*critic_lossl
        
        
        return critic_loss, {
            'critic_loss': critic_loss,
            'critic_lossh': critic_lossh,
            'critic_lossl': critic_lossl,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'target_ql': target_ql.mean(),
            'target_qh': target_qh.mean(),
            'qstd': qstd_o.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    new_target_critic = ema_update(new_critic, critic_tar, tau) if need_ema else critic_tar

    return rng, new_critic, new_target_critic, info



class QDQLearner(Agent):

    name = "qdq"
    model_names = ["actor", "actor_tar", "critic", "critic_tar"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 seed: int,
                 env: str,
                 criticsample: Agent,
                 actor_lr: Union[float, optax.Schedule] = 3e-4,
                 critic_lr: Union[float, optax.Schedule] = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = True,
                 discount: float = 0.99,
                 tau: float = 0.005,  # ema for critic learning
                 update_ema_every: int = 2,
                 lr_decay_steps: int = 2000000,
                 temperature: float = 1.,
                 actor_path: str = None,
                 opt_decay_schedule: str = "cosine",
                 policy_and_target_update_period: int=2,
                 clip_grad_norm: float = 1,
                 num_samples: int =50,
                 alpha:float=0.99,
                 beta:float=0.9,
                 gamma:float=0,
                 **kwargs,
                 ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key,critic_key = jax.random.split(rng, 3)
        act_dim = actions.shape[-1]

        if opt_decay_schedule == "cosine" and lr_decay_steps is not None:
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, lr_decay_steps)
            act_opt = optax.chain(optax.scale_by_adam(),
                                  optax.scale_by_schedule(schedule_fn))
        else:
            act_opt = optax.adam(learning_rate=actor_lr)

                              
        actor_def = NormalTanhPolicy(hidden_dims,
                                     act_dim,
                                     state_dependent_std=False,
                                     dropout_rate=dropout_rate,
                                     log_std_scale=1e-3,
                                     log_std_min=-5.0,
                                     tanh_squash_distribution=False)
                                     
        #actor_def=MSEPolicy(hidden_dims,
                            #act_dim,
                            #dropout_rate=dropout_rate,
                            #layer_norm=layer_norm
                            #)
                                     
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             optimizer=act_opt)

        actor_tar = Model.create(actor_def,
                                 inputs=[actor_key, observations])
        
        
        critic_def = MultiHeadQ(hidden_dims, activations=mish,layer_norm=layer_norm,num_heads=2) #
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              optimizer=optax.adam(learning_rate=critic_lr),
                              clip_grad_norm=clip_grad_norm)
        critic_tar = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions])


        # models
        self.actor = actor
        self.actor_tar = actor_tar
        self.critic = critic
        self.critic_tar = critic_tar
        self.criticsample = criticsample
        

        if actor_path is not None:
            self.load_actor(actor_path)


        self.act_dim = act_dim
        self.discount = discount
        self.tau = tau
        self.policy_and_target_update_period = policy_and_target_update_period

        self.etas = jnp.linspace(0, 0.9, lr_decay_steps)
        
        if 'antmaze' in env:
           self.env=True
           if 'umaze-diverse' in env:
             self.awr=False
           else:
             self.awr=True
        else:
           self.env=False
           self.awr=False

        # training
        self.rng = rng
        self._n_training_steps = 0
        self.update_ema_every = update_ema_every
        self.num_samples=num_samples
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        
        if self.gamma>0:
          self.random_policy=True
        else:
          self.random_policy=False

    def eta_schedule(self):
        if self._n_training_steps < len(self.etas):
            return self.etas[self._n_training_steps]
        return 0.9

    def load_actor(self, actor_path):
        self.actor = self.actor.load(actor_path)
        self.actor_tar = self.actor_tar.load(actor_path)
        print(f"Successfully load pre-trained dbc from {actor_path}")

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1,
                       deterministic: bool = True) -> jnp.ndarray:
                                      
        rng, actions = _sample_actions(self.rng, self.actor.network,
                                      self.actor.params, observations,
                                      temperature, deterministic=True)
                                      
        self.rng = rng

        return jnp.clip(actions,-1,1)

    def update(self, batch: Batch) -> InfoDict:

        info = {}
        # update conditional diffusion bc
        need_ema = self._n_training_steps % self.update_ema_every == 0
        
        self.rng, self.critic, self.critic_tar, new_info = _jit_update_critic(self.rng,
                                                                              self.critic,
                                                                              self.critic_tar,
                                                                              self.actor_tar,
                                                                              self.discount,
                                                                              self.env,
                                                                              self.awr,
                                                                              self.tau,
                                                                              #self.criticsample.get_heun_sampler,
                                                                              self.criticsample.onestep_sampler,
                                                                              batch,
                                                                              #self.num_samples,
                                                                              self.alpha,
                                                                              self.beta,
                                                                              need_ema=need_ema,
                                                                              random_policy=self.random_policy)
        info.update(new_info)
        
        if self._n_training_steps % self.policy_and_target_update_period == 0:
            self.rng, self.actor, self.actor_tar, new_info = jit_update_actor(self.actor,
                                                                              self.actor_tar,
                                                                              self.critic,
                                                                              batch,
                                                                              self.rng,
                                                                              self.env,
                                                                              self.awr,
                                                                              self.tau,
                                                                              self.gamma,
                                                                              need_ema=need_ema)

            info.update(new_info)
        

        self._n_training_steps += 1
        print(self._n_training_steps)
        return info
