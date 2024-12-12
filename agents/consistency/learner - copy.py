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

