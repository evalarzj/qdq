
from functools import partial
from typing import Tuple, Type
import flax.linen as nn
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from utils import sample_n_k
import optax
from networks.model import Model, get_weight_decay_mask
from networks.mlp import MLP
from diffusions.utils import FourierFeatures
from datasets import Batch
from networks.types import InfoDict, Params, PRNGKey
from tqdm import trange, tqdm


@jax.jit
def jit_update_ddpm(actor: Model, batch: Batch, rng: PRNGKey, T, alpha_hat) -> Tuple[Model, InfoDict]:
    rng, t_key, noise_key, tr_key = jax.random.split(rng, 4)
    time = jax.random.randint(t_key, (batch.shape[0],), 0, T)[:, jnp.newaxis]
    eps_sample = jax.random.normal(noise_key, batch.shape)

    noisy_samples = jnp.sqrt(alpha_hat[time]) * batch + jnp.sqrt(1 - alpha_hat[time]) * eps_sample

    def actor_loss_fn(paras: Params) -> Tuple[jnp.ndarray, InfoDict]:
        pred_eps = actor.apply(paras,
                               noisy_samples,
                               time,
                               rngs={'dropout': tr_key},
                               training=True)

        actor_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()

        return actor_loss, {'loss': actor_loss}


    return rng, *actor.apply_gradient(actor_loss_fn)

@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddpm_sampler(noise_pred_apply_fn, params, T, rng, alphas, alpha_hats, betas,
                 sample_temperature,
                 prior,
                 training=False):

    num_samples = prior.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple

        input_time = jnp.expand_dims(jnp.array([time]).repeat(num_samples), axis=1)
        eps_pred = noise_pred_apply_fn(params, current_x, input_time, training=training)

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = ((1 - alphas[time]) / (jnp.sqrt(1 - alpha_hats[time])))
        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(num_samples,) + current_x.shape[1:])
        z_scaled = sample_temperature * z

        # current_x = current_x + (time > 1) * (jnp.sqrt(betas[time]) * z_scaled)
        sigmas_t = jnp.sqrt(betas[time]*(1-alpha_hats[time-1])/(1-alpha_hats[time]))
        current_x = current_x + (time > 1) * (sigmas_t * z_scaled)

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn,
                                   (prior, rng),
                                   jnp.arange(T, 0, -1),
                                   unroll=5)

    return input_tuple


@partial(jax.jit, static_argnames=('noise_pred_apply_fn', 'T', 'repeat_last_step', 'clip_sampler', 'training'))
def ddim_sampler(noise_pred_apply_fn, params, T, rng, alphas, alpha_hats, betas,
                 sample_temperature,
                 prior,
                 ddim_eta=0,
                 training=False):

    # in ddim, if we let ddim_eta = 0, it will magnify the randomness of sampling prior

    num_samples = prior.shape[0]

    def fn(input_tuple, time):
        current_x, rng = input_tuple

        input_time = jnp.expand_dims(jnp.array([time]).repeat(num_samples), axis=1)
        # noise_model(s, a, time, training=training) in DDPM
        eps_pred = noise_pred_apply_fn(params, current_x, input_time, training=training)

        sigmas_t = ddim_eta * jnp.sqrt((1 - alpha_hats[time-1]) / (1 - alpha_hats[time]) * (1 - alphas[time]))

        alpha_1 = 1 / jnp.sqrt(alphas[time])
        alpha_2 = jnp.sqrt(1 - alpha_hats[time])
        alpha_3 = jnp.sqrt(1 - alpha_hats[time-1] - sigmas_t**2)

        current_x = alpha_1 * (current_x - alpha_2 * eps_pred) + alpha_3 * eps_pred

        rng, key = jax.random.split(rng, 2)
        z = jax.random.normal(key, shape=(num_samples,) + current_x.shape[1:])
        z_scaled = sample_temperature * z
        current_x = current_x + sigmas_t * z_scaled

        return (current_x, rng), ()

    key, rng = jax.random.split(rng, 2)
    input_tuple, () = jax.lax.scan(fn,
                                   (prior, rng),
                                   jnp.arange(T, 0, -1),
                                   unroll=5)

    return input_tuple


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class DDPM(nn.Module):
    noise_predictor: Type[nn.Module]
    time_embedding: Type[nn.Module]
    time_processor: Type[nn.Module]

    @nn.compact
    def __call__(self,
                 z: jnp.ndarray,
                 time: jnp.ndarray,
                 training: bool = False):
        t_ff = self.time_embedding()(time)
        time_suffix = self.time_processor()(t_ff, training=training)
        reverse_input = jnp.concatenate([z, time_suffix], axis=-1)
        return self.noise_predictor()(reverse_input, training=training)


data_size = 100
T = 5
training_steps = 50000


rng = jax.random.PRNGKey(0)


# center the whole data!!!!
centers = [jnp.array((0, 0)),
           jnp.array((0, 1)),
           jnp.array((-1, 1)),
           jnp.array((-1, -1)),
           jnp.array((0, -1)),
           jnp.array((1, -1)),
           jnp.array((1, 1))]
scale = 0.2
rng, act_key, *keys = jax.random.split(rng, 2 + len(centers))

mods = [jax.random.normal(k_, (data_size, 2)) for k_ in keys]


dataset = jnp.concatenate([c_ + scale * m_ for c_, m_ in zip(centers, mods)], axis=0)



time_embedding = partial(FourierFeatures,
                         output_size=16,
                         learnable=False)

time_processor = partial(MLP,
                         hidden_dims=(32, 32),
                         activations=mish,
                         activate_final=False)

noise_model = partial(MLP,
                      hidden_dims=(256, 256, 2),
                      activations=mish,
                      activate_final=False)

actor_def = DDPM(time_embedding=time_embedding,
                 time_processor=time_processor,
                 noise_predictor=noise_model)

optimizer = optax.adam(learning_rate=3e-4)

actor = Model.create(actor_def,  # state, action_t, timestep -> action_(t-1) || first dim=batch
                     inputs=[act_key, jnp.zeros((1, 2)), jnp.zeros((1, 1))],
                     optimizer=optimizer)

betas = jnp.linspace(1e-4, 2e-2, T)
betas = jnp.concatenate([jnp.zeros((1,)), betas])
alphas = 1 - betas
alpha_hats = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])


for epoch in trange(training_steps):
    sampled_ids = sample_n_k(dataset.shape[0], 256)
    batch = dataset[sampled_ids]
    rng, actor, info = jit_update_ddpm(actor, batch, rng, T, alpha_hats)
    if epoch % 1000 == 0:
        print(info)

rng, key_normal = jax.random.split(rng, 2)
prior = jax.random.normal(rng, (500, 2))

# plots comparison
# use the same T as training procedure!
pred_samples, rng = ddpm_sampler(actor.apply, actor.params, T, rng, alphas, alpha_hats, betas,
                                 0,  prior)
plt.scatter(dataset[:, 0], dataset[:, 1], s=3)
plt.scatter(pred_samples[:, 0], pred_samples[:, 1], s=3)
plt.show()


# sensitivity of the prior range -> it works
# use zero prior + pure prior define can give more modalities?
x = jnp.linspace(-2, 2, 100)
mean_ = []
sample_pts = []

for x_ in tqdm(x):
    pred_samples, rng = ddim_sampler(actor.apply, actor.params, T, rng, alphas, alpha_hats, betas,
                                     0., jnp.zeros((500, 2)) + jnp.array((x_, 0)))
    #      prior can be jnp.zeros((500, 2))
    mean_.append(pred_samples.mean(axis=0))
    sample_pts.append(pred_samples)
mean_ = jnp.stack(mean_)
sample_pts = jnp.concatenate(sample_pts, axis=0)

plt.plot(x, mean_[:, 0], linewidth=1)
plt.show()

plt.scatter(dataset[:, 0], dataset[:, 1], s=3)
plt.scatter(sample_pts[:, 0], sample_pts[:, 1], s=3)
plt.show()


# grid sampling
grid_pts = jnp.mgrid[-2: 2: 50j, -2: 2: 50j].transpose().reshape((-1, 2))
grid_prior = jax.random.normal(rng, grid_pts.shape) * 0.
grid_pred, rng = ddim_sampler(actor.apply, actor.params, T, rng, alphas, alpha_hats, betas,
                                 0., grid_prior + grid_pts)

plt.scatter(dataset[:, 0], dataset[:, 1], s=3)
plt.scatter(grid_pred[:, 0], grid_pred[:, 1], s=3)
plt.show()

# from diffusers import DDPMScheduler, UNet2DModel
# from PIL import Image
# import torch
# import numpy as np
#
# scheduler = DDPMScheduler.from_pretrained()
# model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
# scheduler.set_timesteps(50)
#
# sample_size = model.config.sample_size
# noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
# input = noise
#
# for t in scheduler.timesteps:
#     with torch.no_grad():
#         noisy_residual = model(input, t).sample
#         prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
#         input = prev_noisy_sample
