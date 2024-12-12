# Q-Distribution guided Q-learning for offline reinforcement learning Uncertainty penalized Q-value via consistency model(QDQ)

Official implementation for NeurIPS 2023 paper [Q-Distribution guided Q-learning for offline reinforcement learning Uncertainty penalized Q-value via consistency model](https://arxiv.org/abs/2410.20312).


## Implementation requirement

1. Install [MuJoCo version 2.1.0](https://github.com/google-deepmind/mujoco/releases?page=2)
2. Install [D4RL](https://github.com/Farama-Foundation/D4RL/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688)
3. Install [jax][https://jax.readthedocs.io/en/latest/installation.html] 


## Run QDQ


### Pretrained consistency model

You can train your own consistency model following command in shell script *run_consistency.sh*. For example, use the following command to train a diffusion model as:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-v2 --agent sm  --max_steps 800000 --save_ckpt True
```

Then use the stored diffusion model to train the consistency model as(change the path to the diffusion model you saved or use the pretrained model we provided):

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-v2 --agent cd --load_Q_ckpt /qdq/results/halfcheetah-medium-v2/20240630-194713_sm --max_steps 1600000 --save_ckpt True 
```

### QDQ learnng

In order to run QDQ, an example command is(the full conmmand is stored in the shell script *run.sh*):

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env halfcheetah-medium-v2 --agent qdq --load_Q_ckpt /qdq/results/halfcheetah-medium-v2/20240630-200724_cd --num_seed_runs 5
```

If you train your own consistency model, then change the path to the model you saved, or you can just use the pretrained model we provided.

The config file is different for each task, pay attention when you train qdq for different task.


## Citation

If you find this code useful for your research, please cite our paper as:

```
@article{zhang2024q,
  title={Q-Distribution guided Q-learning for offline reinforcement learning: Uncertainty penalized Q-value via consistency model},
  author={Zhang, Jing and Fang, Linjiajie and Shi, Kexin and Wang, Wenjia and Jing, Bing-Yi},
  journal={arXiv preprint arXiv:2410.20312},
  year={2024}
}
```



## Acknowledgement

This codebase is built off of the official implementation of IQL (https://github.com/ikostrikov/implicit_q_learning) ,consistency model (https://github.com/openai/consistency_models_cifar10).