import os

import json
import ray
from tensorboardX import SummaryWriter
import numpy as np
from agents import *
from datasets import make_env_and_dataset
from utils import make_env
from utils import prepare_output_dir, MBars
from utils import sample_n_k, sampleqdata, denormalize_value

from absl import app, flags, logging
import time

import jax
import jax.numpy as jnp
import haiku as hk

from agents.scorematch.learner import ScoreLearner
from agents.consistency.learner import ConDistill


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

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

flags.DEFINE_integer('device', 0, 'The device to use.')
flags.DEFINE_string('env', 'hopper-medium-replay-v2', 'Environment name.')
flags.DEFINE_enum('reward_tune', 'no',
                  ['normalize', 'iql_antmaze', 'iql_locomotion', 'cql_antmaze', 'antmaze', 'no'],
                  'Whether to tune the rewards? No tune does not work well')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl'], 'Dataset name.')
flags.DEFINE_enum('agent', 'qd',['qd','cd','sm'], 'Training methods')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_seed_runs', 2, 'number of runs for different seeds')
flags.DEFINE_integer('n_eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10, 'Logging interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 800000, 'Number of training steps.') # set this to 800000 if train score matching model, set to 1500000 if train the consistency model
flags.DEFINE_float('discount', 0.99, 'Discount factor')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('save_ckpt', False, 'Save agents during training.')
flags.DEFINE_boolean('test', False, 'activate test mode. without ray process')
flags.DEFINE_string('tag', '', 'Give a tag to name specific experiment.')



#cd configure
flags.DEFINE_string('load_Q_ckpt', '', 'load Q agents from the dir.')

Learner = {'qd': QDLearner,
           'cd': ConDistill,
           'sm': ScoreLearner}


def _seed_run(learner,
              config: dict,
              dataset,
              fulldata,
              save_dir,
              pbar=None,
              idx=0):
    # record eval stats
    local_seed = config['seed'] + idx
    with open(os.path.join(save_dir, f"seed_{local_seed}.txt"), "w") as f:
        print("\t".join(["steps"] + ["loss_type"]+["loss_value"]), file=f)

    summary_writer = SummaryWriter(os.path.join(save_dir, 'tensorboard', 'seed_' + str(local_seed)))
    video_save_folder = None if not config['save_video'] else os.path.join(
        save_dir, 'video', 'eval')
    ckpt_save_folder = os.path.join(save_dir, 'ckpt')

    env = make_env(config['env'], local_seed, video_save_folder)

    a_config = config.copy()
    a_config['seed'] = local_seed
    a_config['env'] = config['env']
    
    if a_config['agent'] =='qd':
        learnerQ = learner(env.observation_space.sample()[np.newaxis],  # given a batch dim, shape = [1, *(raw_shape)]
                           env.action_space.sample()[np.newaxis],
                           mc_return=np.array([[20]]),
                           mc_return_ori=dataset["mcreturn_ori"],
                           lr_decay_steps=a_config['max_steps'],
                           **a_config)
                           
    if a_config['agent'] =='sm':
        x_con=jnp.concatenate([env.observation_space.sample()[np.newaxis], env.action_space.sample()[np.newaxis]], axis=-1)
        learnerQ = learner(x_con,
                           np.array([[20]]),
                           mc_return_ori=dataset["mcreturn_ori"],
                           lr_decay_steps=a_config['max_steps'],
                           Train=True,
                           train_model=True,
                           **a_config)
    if a_config['agent'] =='cd':
        x_con=jnp.concatenate([env.observation_space.sample()[np.newaxis], env.action_space.sample()[np.newaxis]], axis=-1)
        
        Qlearner = ScoreLearner(x_con,
                                np.array([[20]]),
                                mc_return_ori=dataset["mcreturn_ori"],
                                lr_decay_steps=a_config['max_steps'],
                                Train=False,
                                train_model=False,
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
        
        learnerQ = learner(x_con,
                           np.array([[20]]),
                           mc_return_ori=dataset["mcreturn_ori"],
                           diffuser=Qlearner,
                           lr_decay_steps=config['max_steps'],
                           Train=True,
                           **a_config)
                           
    try:
        for i in range(config['max_steps']):
            batch = sampleqdata(dataset,config['batch_size'])
            update_info = learnerQ.update(batch)
            if i % config['log_interval'] == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        summary_writer.add_scalar(f'training/{k}', v, i)
                    else:
                        summary_writer.add_histogram(f'training/{k}', v, i)
                    with open(os.path.join(save_dir, f"seed_{local_seed}.txt"), "a+") as f:
                        stats = [i] + [k]+[v]
                        print("\t".join([str(_) for _ in stats]), file=f)
                summary_writer.flush()

            if pbar is not None:
                pbar.update.remote(idx)

        # return sampled Q value
        samplekey = jax.random.PRNGKey(local_seed)
        samplekey = hk.PRNGSequence(samplekey)
        
        if a_config['agent'] =='qd':        
            Q_ = learnerQ.sample_Qs(observations=dataset["observations"][0:1000],actions=dataset["actions"][0:1000],rng=next(samplekey),batch_Q=True,denormalize=True)
        if a_config['agent'] =='sm':
           obs_action=jnp.concatenate([dataset["observations"][0:10000], dataset["actions"][0:10000]], axis=-1)
           datatemp=dataset["observations"][0:10000]
           #obs_action=jnp.concatenate([fulldata.observations[0:10000], fulldata.actions[0:10000]], axis=-1)
           #datatemp=fulldata.observations[0:10000]
           datashape=datatemp.shape[0]
           sampling_shape = (datashape*a_config['num_samples'],1)
           Q_ ,n= learnerQ.get_heun_sampler(obs_action,shape=sampling_shape,rng=next(samplekey),denormalize=True)
        if a_config['agent'] =='cd':
            obs_action=jnp.concatenate([dataset["observations"][0:10000], dataset["actions"][0:10000]], axis=-1)
            #obs_action=jnp.concatenate([fulldata.observations[0:10000], fulldata.actions[0:10000]], axis=-1)
            Q_ = learnerQ.onestep_sampler(obs_action,rng=next(samplekey),denormalize=True)
            
        
        # save checkpoints
        if config['save_ckpt']:
            learnerQ.save_ckpt(prefix=f'{idx}_finished_', ckpt_folder=ckpt_save_folder, silence=False)
            with open(os.path.join(save_dir, 'Qminmax.txt'), 'w') as file:
                print(learnerQ.Qmax, file=file)
                print(learnerQ.Qmax)
                print(learnerQ.Qmin, file=file)
                print(learnerQ.Qmin)

        return Q_

    except KeyboardInterrupt:
        # save checkpoints if interrupted
        if config['save_ckpt']:
            learnerQ.save_ckpt(prefix=f'{idx}_expt_', ckpt_folder=ckpt_save_folder, silence=True)


@ray.remote(num_gpus=0.4)
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
    
    qdata=dataset.get_traj_n_mc_return_withsanda(discount=config['discount'], num_sample=100,step=10,tran_len=200,trajnorm=False)
    
    print(qdata["mcreturn_ori"][1:10])
    #print(qdata["mcreturn"][1:10])
    print(len(qdata["mcreturn_ori"]))


    learner = Learner[FLAGS.agent]

    if FLAGS.test:
        print("start testing!")
        # flag_dict = FLAGS.flag_values_dict()
        config['max_steps'] = 1000
        config['seed'] = 123
        final_Q=_seed_run(learner,
                  config,
                  qdata,
                  dataset,
                  save_dir,
                  None,
                  0)
        #f_mean, f_std, f_max, f_min = np.mean(final_Q,axis=0), np.std(final_Q,axis=0), np.max(final_Q,axis=0), np.min(final_Q,axis=0)
        print(f'Final eval: mean={np.mean(final_Q,axis=1)}, std={np.std(final_Q,axis=1)}')
        
        #np.save(os.path.join(save_dir, f"sampled_Q.npy"),final_Q)
        #np.save(os.path.join(save_dir, f"observation.npy"),qdata["observations"])
        #np.save(os.path.join(save_dir, f"action.npy"),qdata["actions"])
        #np.save(os.path.join(save_dir, f"Q.npy"),qdata["mcreturn"])
        
        print("testing passed!")
        return
    pbar = MBars(FLAGS.max_steps, ':'.join([FLAGS.env, FLAGS.agent, FLAGS.tag]), FLAGS.num_seed_runs)

    futures = [seed_run.remote(learner,
                               config,
                               qdata,
                               dataset,
                               save_dir,
                               pbar.process,
                               i)  # send dict to the multi-process
               for i in range(FLAGS.num_seed_runs)]
    pbar.flush()
    final_Q = ray.get(futures)
    #final_Q, final_Q_test = [_[0] for _ in final_res], [_[1] for _ in final_res]
    
    final_Q = np.squeeze(final_Q)
    f_mean, f_std, f_max, f_min = np.mean(final_Q,axis=0), np.std(final_Q,axis=0), np.max(final_Q,axis=0), np.min(final_Q,axis=0)
    print(f'Final eval: mean={np.mean(final_Q,axis=0)}, std={np.std(final_Q,axis=0)}')

    # record the final results of different seeds
    with open(os.path.join(save_dir, f"final_mean_scores.txt"), "w") as f:
        print("\t".join([f"seed_{FLAGS.seed + _}" for _ in range(FLAGS.num_seed_runs)] + ["mean", "std", "max", "min"]),
              file=f)
        print("\t".join([str(_) for _ in final_Q] + [str(f_mean), str(f_std), str(f_max), str(f_min)]),
              file=f)
    
    #save the sampled Q value and original data(training data only)
    np.save(os.path.join(save_dir, f"sampled_Q.npy"),final_Q)
    np.save(os.path.join(save_dir, f"observation.npy"),qdata["observations"])
    np.save(os.path.join(save_dir, f"action.npy"),qdata["actions"])
    np.save(os.path.join(save_dir, f"Q.npy"),qdata["mcreturn_ori"])
    
    
    
    #save the sampled Q value and original data(testing data\whole data)
    #np.save(os.path.join(save_dir, f"sampled_Q_test.npy"),final_Q)
    #np.save(os.path.join(save_dir, f"observation_test.npy"),dataset.observations)
    #np.save(os.path.join(save_dir, f"action_test.npy"),dataset.actions)


if __name__ == '__main__':
    app.run(main)
