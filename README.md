# rl codebase of [Feng Xu](mailto:xufeng@lamda.nju.edu.cn)

## Stable algorithims (Runnable and have good performance):
* [Deep Q Learning](https://arxiv.org/abs/1312.5602) (DQN) 
* [Soft Actor Critic](https://arxiv.org/abs/1801.01290) (SAC)
* [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982) (REDQ)


## Unstable Algorithms (Runnable but have poor performance)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)

## Under Development Algorithms (Unrunnable)
* [Model-based Policy Optimization](https://arxiv.org/abs/1906.08253) (MBPO)

## Usage
``` shell
conda env create -f env.yaml 
conda activate rl_base
source prep
cd "algorithm"
python main.py config_path(config/Hopper.json)  
```