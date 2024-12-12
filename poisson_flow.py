from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.datasets import make_swiss_roll
from matplotlib import animation

# constants
N, D = 2, 1  # number of feature dimensions, expand dimensions
EPS = 1e-5
MAX_Z = 10
N_steps = 1000
dz = MAX_Z/N_steps


# Sample a batch from the swiss roll
def sample_batch(size, noise=1.0):
    x, _ = make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0


# Plot the data set (images with 2 dim)
data = sample_batch(10 ** 4)
# plt.figure(figsize=(16, 12))
# plt.scatter(*data, alpha=0.5, color='red', edgecolor='white', s=40)
# plt.show()
print(data.shape)

y = jnp.concatenate([data, jnp.zeros((len(data), D))], axis=1)


@jax.jit
def electric_field(x, z=dz):
    # shape = (N,)
    assert x.shape[-1] == N
    x = jnp.concatenate([x, jnp.ones(D)*z])

    # x has shape (3,)
    return jnp.sum((y-x)/(jnp.linalg.norm(y-x, axis=1)[:, jnp.newaxis]**(N+D) + EPS), axis=0)


batch_electric_field = jax.vmap(electric_field)


def plot_gradients(data_, xlim=(-1.5, 2.0), ylim=(-1.5, 2.0), nx=50, ny=50, plot_scatter=True, alpha=1.0):
    xx = np.stack(np.meshgrid(np.linspace(*xlim, nx), np.linspace(*ylim, ny)), axis=-1).reshape(-1, 2)
    scores = batch_electric_field(xx)[:, :N]
    scores_norm = jnp.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots

    if plot_scatter:
        plt.figure(figsize=(16, 12))
        plt.scatter(data_[:, 0], data_[:, 1], alpha=0.3, color='red', edgecolor='white', s=40)
        plt.xlim(-1.5, 2.0)
        plt.ylim(-1.5, 2.0)

    quiver = plt.quiver(*xx.T, *scores_log1p.T, width=0.0015, color='black', alpha=alpha)

    return quiver


# plot_gradients(data)
# plt.show()


# single example
@jax.jit
def sample_simple(x):
    # create a step function to pass to jax.scan
    def step(x, z):
        vec = electric_field(x, z)

        vec = vec/(abs(vec[-1]) + EPS)
        x += vec[:N] * dz

        # vec = vec/(jnp.linalg.norm(vec) + EPS)
        # x += vec[:N]
        return x, x

    return jax.lax.scan(step, x, np.arange(MAX_Z, 0, -dz))[1]


x = jnp.array([1.5, 1.5])
samples = sample_simple(x)
plot_gradients(data)
plt.scatter(samples[:, 0], samples[:, 1], color='green', s=20)
# draw arrows for each  step
deltas = samples[1:] - samples[:-1]
# deltas = deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
for i, arrow in enumerate(deltas):
    plt.arrow(samples[i, 0], samples[i, 1], arrow[0], arrow[1], width=1e-4, head_width=2e-2, color="green", linewidth=0.5)

plt.show()




