#  Learn Q distribution

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
from networks.policies import NormalTanhPolicy, sample_actions
from networks.critics import MultiHeadQ
from networks.mlp import MLP
from networks.updates import ema_update
from diffusions.diffusion import DDPM, ddpm_sampler, ddim_sampler
from diffusions.utils import FourierFeatures, cosine_beta_schedule, vp_beta_schedule
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.base import Agent
from utils import sample_n_k, sampleqdata, denormalize_value

"""Implementations of algorithms for continuous control."""


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


@partial(jax.jit,
         static_argnames=('num_samples', 'Q_dim'))
def jit_update_Q(Q: Model,
                 batch: collections.defaultdict,
                 rng: PRNGKey,
                 T,
                 alpha_hat,
                 num_samples: int,
                 Q_dim: int) -> Tuple[PRNGKey, Model, InfoDict]:
    rng, t_key, noise_key, prior_key = jax.random.split(rng, 4)
    batch_size = batch["observations"].shape[0]
    
    t = jax.random.randint(t_key, (batch_size,), 1, T + 1)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch["mcreturn"].shape)
    alpha_1, alpha_2 = jnp.sqrt(alpha_hat[t]), jnp.sqrt(1 - alpha_hat[t])

    noisy_Q = alpha_1 * batch["mcreturn"] + alpha_2 * eps_sample
    
    obs_action=jnp.concatenate([batch["observations"], batch["actions"]], axis=-1)

    rng, tr_key1, tr_key2, tr_key3 = jax.random.split(rng, 4)

    def Q_loss_fn(Q_paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = Q.apply(Q_paras,
                           obs_action,
                           noisy_Q,
                           t,  # t \in range(1, T+1)
                           rngs={'dropout': tr_key1},
                           training=True)
        # behavior cloning loss
        Q_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()

        return Q_loss,{'Q_loss': Q_loss}

    new_Q, info = Q.apply_gradient(Q_loss_fn)

    return rng, new_Q, info
    # return rng, Q.apply_gradient(Q_loss_fn)



@partial(jax.jit, static_argnames=('Q_decoder', 'Q_dim', 'batch_Q', 'num_samples','denormalize'))
def _jit_sample_Q(rng: PRNGKey,
                  diffusion_model: Model,
                  prior: jnp.ndarray,
                  obs_action: jnp.ndarray,
                  Q_dim: int,
                  Qmax: float,
                  Qmin: float,
                  Q_decoder: Callable,
                  batch_Q: bool,
                  num_samples: int,
                  denormalize: bool) -> [PRNGKey, jnp.ndarray]:             
    Qs, rng = Q_decoder(rng, diffusion_model.apply, diffusion_model.params, obs_action, prior)
    Qs = Qs.reshape(-1, num_samples, Q_dim)
    
    if denormalize:
       Qs = denormalize_value(Qs,Qmax,Qmin)

    if batch_Q:
        return rng, Qs

    # used for evaluation
    return rng, Qs[0]


class QDLearner(Agent):
    # Diffusion model for Q distribution learning
    name = "qd"
    model_names = ["Q"]

    def __init__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 mc_return: jnp.ndarray,
                 mc_return_ori: jnp.ndarray,
                 seed: int,
                 Q_lr: Union[float, optax.Schedule] = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 clip_grad_norm: float = 1,
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 discount: float = 0.99,
                 T: int = 5,  # number of backward steps
                 ddim_step: int = 5,
                 num_Q_samples: int = 50,  # number of sampled actions to select from for action
                 num_last_repeats: int = 0,
                 clip_sampler: bool = True,
                 time_dim: int = 16,
                 beta_schedule: str = 'vp',
                 lr_decay_steps: int = 2000000,
                 sampler: str = "ddim",
                 Q_prior: str = 'normal',
                 temperature: float = 1.,
                 Q_path: str = None,
                 Train:bool=True,
                 Qmin:float=0.0,
                 Qmax:float=1.0,
                 **kwargs,
                 ):

        rng = jax.random.PRNGKey(seed)
        rng, Q_key = jax.random.split(rng, 2)
        Q_dim = mc_return.shape[-1]
        obs_action=jnp.concatenate([observations, actions], axis=-1)

        # define behavior diffusion model
        time_embedding = partial(FourierFeatures,
                                 output_size=time_dim,
                                 learnable=False)

        time_processor = partial(MLP,
                                 hidden_dims=(32, 32),
                                 activations=mish,
                                 activate_final=False)

        if lr_decay_steps is not None:
            Q_lr = optax.cosine_decay_schedule(Q_lr, lr_decay_steps)

        noise_model = partial(MLP,
                              hidden_dims=tuple(list(hidden_dims) + [Q_dim]),
                              activations=mish,
                              layer_norm=layer_norm,
                              dropout_rate=dropout_rate,
                              activate_final=False)

        Q_def = DDPM(time_embedding=time_embedding,
                         time_processor=time_processor,
                         noise_predictor=noise_model)

        Q = Model.create(Q_def,
                             inputs=[Q_key, obs_action,mc_return, jnp.zeros((1, 1))],  # time
                             optimizer=optax.adam(learning_rate=Q_lr),
                             clip_grad_norm=clip_grad_norm)

        # models
        self.Q = Q
        

        if Q_path is not None:
            self.load_Q(Q_path)

        # sampler
        self.Q_prior = Q_prior
        self.sampler = sampler
        self.temperature = temperature

        if beta_schedule == 'cosine':
            self.betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            self.betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            self.betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')
        # add a special beginning beta[0] = 0 so that alpha[0] = 1, it is used for ddim
        self.betas = jnp.concatenate([jnp.zeros((1,)), self.betas])

        alphas = 1 - self.betas
        self.alphas = alphas
        self.alpha_hats = jnp.cumprod(alphas)
        self.sqrt_alpha_hat_T = jnp.sqrt(self.alpha_hats[-1])

        assert T >= ddim_step, f"timestep {T} should >= ddim_step {ddim_step}"
        self.T = T
        self.ddim_step = ddim_step

        self.num_Q_samples = num_Q_samples
        self.num_last_repeats = num_last_repeats
        self.Q_dim = Q_dim
        self.clip_sampler = clip_sampler
        self.discount = discount
        
        self.Train=Train
        
        #maximium and minimium for the original mc_return(used for scale the Q value)
        if self.Train:
           self.Qmax=jnp.max(mc_return_ori, axis=0)
           self.Qmin=jnp.min(mc_return_ori, axis=0)
        else:
           self.Qmax=Qmax
           self.Qmin=Qmin
        


        self.etas = jnp.linspace(0, 0.9, lr_decay_steps)

        # training
        self.rng = rng
        self._n_training_steps = 0


    def eta_schedule(self):
        if self._n_training_steps < len(self.etas):
            return self.etas[self._n_training_steps]
        return 0.9

    def load_Q(self, Q_path):
        self.Q = self.Q.load(Q_path)
        print(f"Successfully load pre-trained dbc from {Q_path}")

    def Q_decoder(self, key: PRNGKey, model_apply_fn: Callable, params: Params, obs_action: jnp.array, prior: jnp.array):
        if self.sampler == 'ddim':
            return ddim_sampler(key, model_apply_fn, params, self.T, obs_action,self.alphas,
                                self.alpha_hats, self.temperature, self.num_last_repeats, self.clip_sampler, prior,
                                self.ddim_step, guidance_scale=0)

        elif self.sampler == 'ddpm':
            return ddpm_sampler(key, model_apply_fn, params, self.T, obs_action, self.alphas,
                                self.alpha_hats, self.temperature, self.num_last_repeats, self.clip_sampler, prior)

        else:
            raise ValueError(f"{self.sampler} not in ['ddpm', 'ddim']")
    

    def ncsn(self, x_con:jnp.array, x: jnp.array, t: jnp.array):
        eps_pred = self.Q.apply(self.Q.params, x_con,x, t, training=False)
        t_ceil=jnp.ceil(t)
        t_ceil=t_ceil.astype(int)
        alpha_2 = (-1 / (jnp.sqrt(1 - self.alpha_hats[t_ceil])))
        #print(eps_pred)
        score=alpha_2*eps_pred
        return score     
        

    def update(self, batch: collections.defaultdict) -> InfoDict:

        info = {}
        # update conditional diffusion bc
        self.rng, self.Q, new_info = jit_update_Q(self.Q,
                                                  batch,
                                                  self.rng,
                                                  self.T,
                                                  self.alpha_hats,
                                                  1,
                                                  self.Q_dim)  # small stuck

        info.update(new_info)

        self._n_training_steps += 1
        print(self._n_training_steps)
        return info

    def sample_Qs(self,
                  observations: np.ndarray,
                  actions: np.ndarray,
                  rng:PRNGKey,
                  temperature=None,
                  batch_Q=False,
                  denormalize=False) -> jnp.ndarray:

        # set num_repeats = 1, it's just conditional generation of diffusions
        if len(observations.shape) == 1:
            observations = observations[jnp.newaxis, :]  # batch of actions  (B, dim(obs))
        
        if len(actions.shape) == 1:
            actions = actions[jnp.newaxis, :]  # batch of actions  (B, dim(obs))
            
        obs_action=jnp.concatenate([observations, actions], axis=-1)
        obs_action = jax.device_put(obs_action)
        obs_action = obs_action.repeat(self.num_Q_samples, axis=0)  # (B*num_samples, dim_obs)
        
        if self.Train:
           self.rng,key = jax.random.split(self.rng)
        else:
           self.rng, key = jax.random.split(rng)
           #self.rng=rng
           #key=rng
        
        
        
        if self.Q_prior == 'zeros':
            prior = jnp.zeros((obs_action.shape[0], self.Q_dim))
        elif self.Q_prior == 'normal':
            prior = jax.random.normal(key, (obs_action.shape[0], self.Q_dim))
        else:
            raise ValueError(f"self.Q_prior={self.Q_prior} not implemented")
        

        self.rng, Q = _jit_sample_Q(self.rng,
                                    self.Q,
                                    prior,
                                    obs_action,
                                    self.Q_dim,
                                    self.Qmax,
                                    self.Qmin,
                                    self.Q_decoder,
                                    batch_Q=batch_Q,
                                    num_samples=self.num_Q_samples,
                                    denormalize=denormalize)

        #return np.asarray(Q)
        return Q


# if __name__ == '__main__':

    # import numpy as np
    # from datasets import make_env_and_dataset

    # env, dataset = make_env_and_dataset('halfcheetah-medium-v2', 23, 'd4rl',reward_tune='iql_locomotion')
    # # dataset.mc_return = dataset.get_future_mc_return(discount=0.99, avg_credit_assign=False)
    # # dataset.mc_return = dataset.normalize_mc_return(100)
    # #print(dataset)
    
    # qdata=dataset.get_traj_n_mc_return_withsanda(discount=0.99, num_sample=50)
    
    
    # print(qdata["mcreturn_ori"][1:10])

    # learner = QDLearner(seed=32,
                         # observations=env.observation_space.sample()[np.newaxis],
                         # actions=env.action_space.sample()[np.newaxis],
                         # mc_return=np.array([[20]]),
                         # mc_return_ori=qdata["mcreturn_ori"],
                         # sampler='ddpm')

    # state = env.observation_space.sample()[np.newaxis]
    # act_T = env.action_space.sample()[np.newaxis]
    # t = jnp.ones((1, 1))
    
    # obs_action=jnp.concatenate([state, act_T], axis=-1)

    # pred_noise = learner.Q(obs_action, np.array([[20]]), t)
    
    # print(pred_noise)

    # batch = sampleqdata(qdata,5)

  
    # for i in range(500):
        # train_stat = learner.update(batch)
        # for k_, v_ in train_stat.items():
            # print(k_, v_)
    

    # Q_ = learner.sample_Qs(observations=batch["observations"],actions=batch["actions"],batch_Q=True,denormalize=True).reshape(5,-1)
    # #Q_1 = learner.sample_Qs(observations=batch["observations"][1],actions=batch["actions"][1],batch_Q=True)
    # #Q_2 = learner.sample_Qs(observations=batch["observations"][2],actions=batch["actions"][2],batch_Q=True)
    
    # #print("Sampled Q!")
    # print(Q_.shape)
    
    # print("Sampled unnorm Q0!")
    # #print(Q_)
    # print("Sampled unnorm Q1!")
    # #print(np.mean(denormalize_value(Q_1,qdata)))
    # print("Sampled unnorm Q2!")
    # #print(np.mean(denormalize_value(Q_2,qdata)))
   
