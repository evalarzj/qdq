

########halfcheetah-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env halfcheetah-medium-v2 --agent qdq --load_Q_ckpt /qdq/results/halfcheetah-medium-v2/20240630-200724_cd --num_seed_runs 5


########halfcheetah-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env halfcheetah-medium-expert-v2 --agent qdq --load_Q_ckpt /qdq/results/halfcheetah-medium-expert-v2/20240630-211656_cd --max_steps 1000000 --num_seed_runs 5

########halfcheetah-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env halfcheetah-medium-replay-v2 --agent qdq --load_Q_ckpt /qdq/results/halfcheetah-medium-replay-v2/20240630-211740_cd  --num_seed_runs 5

########hopper-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env hopper-medium-v2 --agent qdq --load_Q_ckpt /qdq/results/hopper-medium-v2/20240630-222459_cd --num_seed_runs 5

########hopper-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env hopper-medium-expert-v2 --agent qdq --load_Q_ckpt /qdq/results/hopper-medium-expert-v2/20240630-222534_cd --max_steps 1000000 --num_seed_runs 5

########hopper-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env hopper-medium-replay-v2 --agent qdq --load_Q_ckpt /qdq/results/hopper-medium-replay-v2/20240630-234606_cd --max_steps 1000000 --num_seed_runs 5

########walker2d-medium###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env walker2d-medium-v2 --agent qdq --load_Q_ckpt /qdq/results/walker2d-medium-v2/20240630-234658_cd --num_seed_runs 5

########walker2d-medium-replay###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env walker2d-medium-replay-v2 --agent qdq --load_Q_ckpt /qdq/results/walker2d-medium-replay-v2/20240701-094011_cd --num_seed_runs 5


########walker2d-medium-expert###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env walker2d-medium-expert-v2 --agent qdq --load_Q_ckpt /qdq/results/walker2d-medium-expert-v2/20240701-094046_cd --num_seed_runs 5

########antmaze-umaze###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-umaze-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-umaze-v0/20240701-104827_cd --num_seed_runs 5  --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000


########antmaze-umaze-diverse###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-umaze-diverse-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-umaze-diverse-v0/20240701-111817_cd --num_seed_runs 5 --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000

########antmaze-medium-play###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-medium-play-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-medium-play-v0/20240701-132351_cd --num_seed_runs 5  --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000

########antmaze-medium-diverse###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-medium-diverse-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-medium-diverse-v0/20240701-135143_cd --num_seed_runs 5  --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000


########antmaze-large-play###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-large-play-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-large-play-v0/20240701-143531_cd --num_seed_runs 5  --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000


########antmaze-large-diverse###########
XLA_PYTHON_CLIENT_PREALLOCATE=false python3.9  main.py --env antmaze-large-diverse-v0 --agent qdq --reward_tune iql_antmaze --load_Q_ckpt /qdq/results/antmaze-large-diverse-v0/20240701-145528_cd --num_seed_runs 5  --n_eval_episodes 100 --eval_interval 100000 --max_steps 1000000