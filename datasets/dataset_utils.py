from typing import Tuple

import gym

# from jaxrl.datasets.awac_dataset import AWACDataset
from datasets.d4rl_dataset import D4RLDataset
from datasets.dataset import Dataset
import numpy as np
import d4rl
# from jaxrl.datasets.rl_unplugged_dataset import RLUnpluggedDataset
from utils import iql_normalize
import wrappers


def make_env_and_dataset(env_name: str, seed: int, dataset_name: str,
                         video_save_folder: str = None, reward_tune: bool = 'no',
                         episode_return: bool = False) -> Tuple[gym.Env, Dataset]:
    # env = make_env(env_name, seed, video_save_folder)
    env = gym.make(env_name)  # test env. only
    env = wrappers.EpisodeMonitor(env)  # record info['episode']['return', 'length', 'duration']
    env = wrappers.SinglePrecision(env)  # -> np.float32

    if video_save_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_save_folder)

    # set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    # reward normalization
    if reward_tune == 'normalize':
        dataset.rewards = (dataset.rewards - dataset.rewards.mean()) / dataset.rewards.std()
    elif reward_tune == 'iql_antmaze':
        dataset.rewards = dataset.rewards - 1.0
    elif reward_tune == 'iql_locomotion':
        dataset.rewards = iql_normalize(dataset)
    elif reward_tune == 'cql_antmaze':
        dataset.rewards = (dataset.rewards - 0.5) * 4.0
    elif reward_tune == 'antmaze':
        dataset.rewards = (dataset.rewards - 0.25) * 2.0

    # get MC returns?
    if episode_return:
        dataset.episode_returns = env.get_normalized_score(dataset.get_episode_returns()) * 100.

    # if 'antmaze' in env_name:
    #     dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    # elif 'halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name:

        # get human-normalized returns, just for visualization only
        # dataset.episode_returns = env.get_normalized_score(dataset.get_episode_returns()) * 100.

        # normalize rewards: two values are not compatible !!
        # iql_normalize(dataset)

    # set mc-return
    # dataset.mc_return = dataset.get_future_mc_return(discount=0.99)

    assert 'd4rl' in dataset_name, "Only support d4rl dataset right now"

    return env, dataset
    

