"""
Used for single process testing
"""


import os
import numpy as np
from tensorboardX import SummaryWriter

from agents import *
from datasets import make_env_and_dataset
from utils import make_env
from eval import eval_agents, STATISTICS
from utils import prepare_output_dir, MBars
from plots import plot_curve
from tqdm import tqdm

from absl import app, flags

# from ml_collections import config_flags

FLAGS = flags.FLAGS
# 'walker2d-expert-v2'  'halfcheetah-expert-v2' 'ant-medium-v2'
flags.DEFINE_string('env', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl'], 'Dataset name.')
flags.DEFINE_enum('reward_tune', 'iql_locomotion',
                  ['normalize', 'iql_antmaze', 'iql_locomotion', 'cql_antmaze', 'antmaze'],
                  'Whether to tune the rewards.')
flags.DEFINE_enum('agent', 'dpi',
                  ['bc', 'hql', 'iql', 'sac', 'dbc', 'qcd', 'dql', 'dpi'],
                  'Training methods')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_seed_runs', 5, 'number of runs for different seeds')
flags.DEFINE_integer('n_eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_float('percentile', 100.0, 'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0, 'Percentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_string('tag', 'test', 'Give a tag to name specific experiment.')

# dpi
# flags.DEFINE_integer('num_samples', 10, 'Number of sampled x_0 for Q-learning')
# flags.DEFINE_integer('num_action_samples', 50, 'Number of sampled action candidates')
# flags.DEFINE_enum('action_prior', 'normal', ['normal', 'zeros'],
#                   'The prior to sample actions for evaluation')

# pcd config
# flags.DEFINE_integer('num_cond_classes', 20,  'Number of condition classes. = 0 if continuous')
# flags.DEFINE_integer('cond_feature_dim', 16,  'Dimension of conditional embedding')
# flags.DEFINE_integer('num_samples', 1,  'Number of sampled x_0 for Q-learning')
# flags.DEFINE_integer('num_action_samples', 10,  'Number of sampled action candidates')
# flags.DEFINE_enum('action_prior', 'normal',  ['normal', 'zeros'],
#                   'The prior to sample actions for evaluation')
# flags.DEFINE_boolean('cond_embed_learnable', False,  'Whether the embedding is learnable')
# flags.DEFINE_float('guidance_scale', 0.,  'guidance scale = (1+w) in CFG')
# flags.DEFINE_float('guidance_rescale', 0.7,  'guidance scale = (1+w) in CFG')
# flags.DEFINE_enum('condition', 'future_mc', ['traj_mc', 'future_mc'],
#                   'Determine the initial credit assignment of transitions.')
# flags.DEFINE_boolean('norm_mc', True,  'Whether normalize data mc return to [0, 1] or get the label?')


def main(_):
    config = __import__(f'configs.{FLAGS.agent}_config', fromlist=('configs', FLAGS.agent)).config
    save_dir = prepare_output_dir(folder=os.path.join('results', FLAGS.env),
                                  time_stamp=True,
                                  suffix=FLAGS.agent + FLAGS.tag)
    print(f"\nSave results to: {save_dir}\n")

    # save config:
    print('=' * 10 + ' Arguments ' + '=' * 10)
    with open(os.path.join(save_dir, 'config.txt'), 'w') as file:
        for k, v in config.items():
            if hasattr(FLAGS, k):
                value = str(getattr(FLAGS, k)) + "*"  # re-claimed in FLAG definition
            else:
                value = str(v)
            print(k + ' = ' + value)
            print(k + ' = ' + value, file=file)

    config.update(FLAGS.flag_values_dict())

    Learner = {'bc': BCLearner,
               'iql': IQLLearner,
               'sac': SACLearner,
               'ivr': IVRLearner,
               'hql': HQLLearner,
               'dbc': DDPMBCLearner,
               'qcd': QCDLearner,
               'dpi': DPILearner,
               'dql': DQLLearner}[FLAGS.agent]

    with open(os.path.join(save_dir, f"seed_{FLAGS.seed}.txt"), "w") as f:
        print("\t".join(["steps"] + STATISTICS), file=f)

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard', str(FLAGS.seed)))
    ckpt_save_folder = os.path.join(save_dir, 'ckpt')
    video_save_folder = None if not FLAGS.save_video else os.path.join(
        save_dir, 'video', 'eval')

    env, dataset = make_env_and_dataset(FLAGS.env, FLAGS.seed, FLAGS.dataset_name, video_save_folder,
                                        reward_tune=config["reward_tune"])

    # env = make_env(FLAGS.env, FLAGS.seed, video_save_folder)

    if FLAGS.percentage < 100.0:
        dataset.take_random(FLAGS.percentage)

    if FLAGS.percentile < 100.0:
        dataset.take_top(FLAGS.percentile)

    # set mc-return using discount as in config
    # num_cond_classes gives 0-(N-1) N int value for class indices
    # set mc-return
    # if FLAGS.condition == 'traj_mc':
    #     dataset.mc_return = dataset.get_traj_wise_mc_return(discount=config['discount'], avg_credit_assign=True)
    # elif FLAGS.condition == 'future_mc':
    #     dataset.mc_return = dataset.get_future_mc_return(discount=config['discount'], avg_credit_assign=True)
    # else:
    #     raise ValueError("FLAGS.condition")
    #
    # if FLAGS.norm_mc:
    #     dataset.mc_return = dataset.normalize_mc_return(config['num_prompt_classes'])

    agent = Learner(env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    lr_decay_steps=FLAGS.max_steps,
                    **config)

    try:
        max_mean_return = float("-inf")

        for i in tqdm(range(FLAGS.max_steps), desc=f'Train {FLAGS.env}:{FLAGS.agent}'):

            if i % FLAGS.eval_interval == 0:
                agent.save_ckpt(prefix="test_eval_", ckpt_folder=ckpt_save_folder, silence=True)
                eval_stat = eval_agents(i, agent, env, summary_writer, save_dir, FLAGS.seed, FLAGS.n_eval_episodes)
                if eval_stat['mean'] > max_mean_return:
                    agent.save_ckpt(prefix=f"{i}test_best_", ckpt_folder=ckpt_save_folder, silence=True)
                    max_mean_return = eval_stat['mean']

            batch = dataset.sample(FLAGS.batch_size)
            update_info = agent.update(batch)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                summary_writer.flush()

        # return final evaluations
        eval_agents(FLAGS.max_steps, agent, env, summary_writer, save_dir, FLAGS.seed, FLAGS.n_eval_episodes)

        agent.save_ckpt(prefix="test_finished_", ckpt_folder=ckpt_save_folder, silence=True)

    except KeyboardInterrupt:
        agent.save_ckpt(prefix="test_expt_", ckpt_folder=ckpt_save_folder, silence=True)

    # fig, ax = plot_curve(dirname=save_dir, label=FLAGS.agent)
    # fig.show()


if __name__ == '__main__':
    app.run(main)
