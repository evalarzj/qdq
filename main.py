import os

import json
import ray
from tensorboardX import SummaryWriter
import numpy as np
from agents import *
from datasets import make_env_and_dataset
from utils import make_env
from utils import prepare_output_dir, MBars
from eval import eval_agents, STATISTICS
from plots import plot_curve
from collections import deque

from absl import app, flags, logging

import jax
import jax.numpy as jnp


from agents.consistency.learner import ConDistill
from agents.scorematch.learner import ScoreLearner

os.environ["CUDA_VISIBLE_DEVICES"] = "5,4"
ray.init()
# ray.init(num_gpus=8)
# ray.init(num_gpus=8,
#          _system_config={
#              "object_spilling_config": json.dumps(
#                  {"type": "filesystem", "params": {"directory_path": "/mnt/sdb/jj/tmp/ray"}},
#              )
#          }, )
# ray.init(num_gpus=5)

FLAGS = flags.FLAGS
# 'walker2d-expert-v2'  'halfcheetah-expert-v2' 'ant-medium-v2'    hopper-medium-v2
flags.DEFINE_integer('device', 0, 'The device to use.')
flags.DEFINE_string('env', 'hopper-medium-replay-v2', 'Environment name.')
flags.DEFINE_enum('reward_tune', 'no',
                  ['normalize', 'iql_antmaze', 'iql_locomotion', 'cql_antmaze', 'antmaze', 'no'],
                  'Whether to tune the rewards? No tune does not work well')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl'], 'Dataset name.')
flags.DEFINE_enum('agent', 'qdq',
                  ['bc', 'iql', 'sac', 'ivr', 'hql', 'dbc', 'qcd', 'dql', 'dpi','qdq'],
                  'Training methods')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_seed_runs', 2, 'number of runs for different seeds')
flags.DEFINE_integer('n_eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_float('percentile', 100.0, 'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0, 'Percentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_ckpt', False, 'Save agents during training.')
flags.DEFINE_boolean('test', False, 'activate test mode. without ray process')
flags.DEFINE_string('tag', '', 'Give a tag to name specific experiment.')

#qdq configure
flags.DEFINE_string('load_Q_ckpt', '', 'load Q agents from the dir.')


Learner = {'bc': BCLearner,
           'iql': IQLLearner,
           'sac': SACLearner,
           'ivr': IVRLearner,
           'hql': HQLLearner,
           'dbc': DDPMBCLearner,
           'qcd': QCDLearner,
           'dpi': DPILearner,
           'dql': DQLLearner,
           'qdq': QDQLearner}


def _seed_run(learner,
              config: dict,
              dataset,
              save_dir,
              pbar=None,
              idx=0):
    # record eval stats
    local_seed = config['seed'] + idx
    with open(os.path.join(save_dir, f"seed_{local_seed}.txt"), "w") as f:
        print("\t".join(["steps"] + STATISTICS), file=f)

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard', 'seed_' + str(local_seed)))
    video_save_folder = None if not config['save_video'] else os.path.join(
        save_dir, 'video', 'eval')
    ckpt_save_folder = os.path.join(save_dir, 'ckpt')

    env = make_env(config['env'], local_seed, video_save_folder)
    
    a_config = config.copy()
    a_config['seed'] = local_seed

    if config['percentage'] < 100.0:
        dataset.take_random(config['percentage'])

    if config['percentile'] < 100.0:
        dataset.take_top(config['percentile'])
        
    x_con=jnp.concatenate([env.observation_space.sample()[np.newaxis], env.action_space.sample()[np.newaxis]], axis=-1)
    Qlearner =ConDistill(x_con,
                         np.array([[20]]),
                         np.array([[20]]),
                         #diffuser=Qlearner,
                         lr_decay_steps=config['max_steps'],
                         Train=False,
                         **a_config)
    
    
    if config['load_Q_ckpt']:
        Qminmax=[]
        ckpt_folder=os.path.join(config['load_Q_ckpt'],"ckpt")
        Qlearner.load_ckpt(prefix="0_finished_",ckpt_folder=ckpt_folder, silence=False)
        with open(os.path.join(config['load_Q_ckpt'], 'Qminmax.txt'), 'r') as file:
            for line in file:
               tmp=line.strip().replace('[','').replace(']','')
               Qminmax.append(float(tmp))
        Qlearner.Qmax=Qminmax[0]
        Qlearner.Qmin=Qminmax[1]
        print("!!!!!!!!!!!!!!!!!!!!!!!")
        print(Qlearner.Qmax)
        print(Qlearner.Qmin)
    else:
        raise ValueError("The path for loading Q distribution model is empty or invalid")
    

    
    agent = learner(env.observation_space.sample()[np.newaxis],  # given a batch dim, shape = [1, *(raw_shape)]
                    env.action_space.sample()[np.newaxis],
                    criticsample=Qlearner,
                    lr_decay_steps=config['max_steps'],
                    **a_config)

    last_window_mean_return = deque(maxlen=5)

    try:
        running_max_return = float("-inf")
        for i in range(config['max_steps']):

            if i % config['eval_interval'] == 0:
                eval_res = eval_agents(i, agent, env, summary_writer, save_dir, local_seed,
                                       config['n_eval_episodes'])
                last_window_mean_return.append(eval_res['mean'])
                
                if eval_res['mean'] > running_max_return:
                    running_max_return = eval_res['mean']
                    if config['save_ckpt']:
                        agent.save_ckpt(prefix=f'{idx}_best_', ckpt_folder=ckpt_save_folder,  silence=False)
                    #print('meet best')
                if config['save_ckpt']:
                    agent.save_ckpt(prefix=f'{idx}_eval_', ckpt_folder=ckpt_save_folder,  silence=True)

            batch = dataset.sample(config['batch_size'])
            
           
            
            update_info = agent.update(batch)
            
            if i % config['log_interval'] == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                    #with open(os.path.join(save_dir, f"seed_{local_seed}111.txt"), "a+") as f:
                        #stats = [i] + [k]+[v]
                        #print("\t".join([str(_) for _ in stats]), file=f)
                summary_writer.flush()

            if pbar is not None:
                pbar.update.remote(idx)

        # return final evaluations
        final_eval = eval_agents(config['max_steps'], agent, env, summary_writer, save_dir, local_seed,
                                 config['n_eval_episodes'])
        
        

        last_window_mean_return.append(final_eval['mean'])
        
        
        

        # save checkpoints
        if config['save_ckpt']:
            agent.save_ckpt(prefix=f'{idx}_finished_', ckpt_folder=ckpt_save_folder, silence=False)

        return np.mean(last_window_mean_return), running_max_return

    except KeyboardInterrupt:
        # save checkpoints if interrupted
        if config['save_ckpt']:
            agent.save_ckpt(prefix=f'{idx}_expt_', ckpt_folder=ckpt_save_folder, silence=True)


@ray.remote(num_gpus=0.3)
def seed_run(*args, **kwargs):
    return _seed_run(*args, **kwargs)


def main(_):
    config = __import__(f'configs.{FLAGS.agent}_config', fromlist=('configs', FLAGS.agent)).config
    save_dir = prepare_output_dir(folder=os.path.join('/home/jzhanggy/offlineRL-main/results', FLAGS.env),
                                  time_stamp=True,
                                  suffix=FLAGS.agent + FLAGS.tag)
    print(f"\nSave results to: {save_dir}\n")

    # update config if is specified in FLAG

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

    _, dataset = make_env_and_dataset(FLAGS.env, FLAGS.seed, FLAGS.dataset_name, reward_tune=config["reward_tune"])
    

    learner = Learner[FLAGS.agent]
    
    #env = make_env(FLAGS.env, FLAGS.seed)
    
   

    if FLAGS.test:
        print("start testing!")
        # flag_dict = FLAGS.flag_values_dict()
        config['max_steps'] = 500000
        config['eval_interval'] = 5000
        config['log_interval'] = 1000
        config['seed'] = 123
        _seed_run(learner,
                  config,
                  dataset,
                  save_dir,
                  None,
                  0)
        print("testing passed!")
        return
    pbar = MBars(FLAGS.max_steps, ':'.join([FLAGS.env, FLAGS.agent, FLAGS.tag]), FLAGS.num_seed_runs)

    futures = [seed_run.remote(learner,
                               config,
                               dataset,
                               save_dir,
                               pbar.process,
                               i)  # send dict to the multi-process
               for i in range(FLAGS.num_seed_runs)]
    pbar.flush()
    final_res = ray.get(futures)
    final_scores, running_best = [_[0] for _ in final_res], [_[1] for _ in final_res]
    f_mean, f_std, f_max, f_min = np.mean(final_scores), np.std(final_scores), max(final_scores), min(final_scores)
    b_mean, b_std, b_max, b_min = np.mean(running_best), np.std(running_best), max(running_best), min(running_best)
    print(f'Final eval: mean={np.mean(final_scores)}, std={np.std(final_scores)}')

    # record the final results of different seeds
    with open(os.path.join(save_dir, f"final_mean_scores.txt"), "w") as f:
        print("\t".join([f"seed_{FLAGS.seed + _}" for _ in range(FLAGS.num_seed_runs)] + ["mean", "std", "max", "min"]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in final_scores] + [str(f_mean), str(f_std), str(f_max), str(f_min)]),
              file=f)
        print("\t".join([str(round(_, 2)) for _ in running_best] + [str(b_mean), str(b_std), str(b_max), str(b_min)]),
              file=f)

    fig, ax = plot_curve(save_dir, label=":".join([FLAGS.agent, FLAGS.env]))
    fig.savefig(os.path.join(save_dir, "training_curve.png"))


if __name__ == '__main__':
    app.run(main)
