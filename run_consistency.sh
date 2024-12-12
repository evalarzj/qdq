

########halfcheetah-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-v2 --agent sm  --max_steps 800000 --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-v2 --agent cd --load_Q_ckpt /qdq/results/halfcheetah-medium-v2/20240630-194713_sm --max_steps 1600000 --save_ckpt True 


########halfcheetah-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-expert-v2 --agent sm  --max_steps 800000 --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-expert-v2 --agent cd --load_Q_ckpt /qdq/results/halfcheetah-medium-expert-v2/20240630-210056_sm --max_steps 1600000 --save_ckpt True

########halfcheetah-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-replay-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env halfcheetah-medium-replay-v2 --agent cd --load_Q_ckpt /qdq/results/halfcheetah-medium-replay-v2/20240630-210222_sm --max_steps 1000000  --save_ckpt True


########hopper-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-v2 --agent sm  --max_steps 800000  --save_ckpt True  

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-v2 --agent cd --load_Q_ckpt /qdq/results/hopper-medium-v2/20240630-220704_sm  --max_steps 1600000 --save_ckpt True

########hopper-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-expert-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-expert-v2 --agent cd --load_Q_ckpt /qdq/results/hopper-medium-expert-v2/20240630-220749_sm --max_steps 1600000 --save_ckpt True

########hopper-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-replay-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env hopper-medium-replay-v2 --agent cd --load_Q_ckpt /qdq/results/hopper-medium-replay-v2/20240630-231431_sm --max_steps 1600000  --save_ckpt True


########walker2d-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-v2 --agent cd --load_Q_ckpt /qdq/results/walker2d-medium-v2/20240630-231505_sm  --max_steps 1600000 --save_ckpt True


########walker2d-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-replay-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-replay-v2 --agent cd --load_Q_ckpt /qdq/results/walker2d-medium-replay-v2/20240701-091653_sm  --max_steps 1600000 --save_ckpt True


########walker2d-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-expert-v2 --agent sm  --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env walker2d-medium-expert-v2 --agent cd --load_Q_ckpt /qdq/results/walker2d-medium-expert-v2/20240701-091717_sm  --max_steps 2000000 --save_ckpt True

########antmaze-umaze###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-umaze-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-umaze-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-umaze-v0/20240701-102325_sm  --max_steps 1600000 --save_ckpt True

########antmaze-umaze-diverse###########
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-umaze-diverse-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-umaze-diverse-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-umaze-diverse-v0/20240701-104900_sm  --max_steps 1600000 --save_ckpt True

########antmaze-medium-play###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-medium-play-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-medium-play-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-medium-play-v0/20240701-114835_sm  --max_steps 1600000 --save_ckpt True

########antmaze-medium-diverse###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-medium-diverse-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-medium-diverse-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-medium-diverse-v0/20240701-132418_sm  --max_steps 1600000 --save_ckpt True


########antmaze-large-play###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-large-play-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-large-play-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-large-play-v0/20240701-142043_sm  --max_steps 1600000 --save_ckpt True


########antmaze-large-diverse###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-large-diverse-v0 --agent sm --reward_tune iql_antmaze --max_steps 800000  --save_ckpt True

XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main_Q.py --env antmaze-large-diverse-v0 --agent cd --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-large-diverse-v0/20240701-143552_sm  --max_steps 1600000 --save_ckpt True
