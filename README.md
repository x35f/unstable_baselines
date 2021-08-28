# Reinforcement Learning Codebase of [Feng Xu](mailto:xufeng@lamda.nju.edu.cn)

Unstable Baselines is designed to be a quick-start guide for Reinforcement Learning. Only the basic version of each algorithm is implemented, without tedious training tricks and tools, e.g., Multithread Sampling and Multi-GPU training. Unstable baselines is beginner-friendly and the implementations should be enough for basic research purpose.


---
## Stable algorithims (Runnable and have good performance):
* [Deep Q Learning](https://arxiv.org/abs/1312.5602) (DQN) 
* [Soft Actor Critic](https://arxiv.org/abs/1801.01290) (SAC)
* [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982) (REDQ)


## Unstable Algorithms (Runnable but have poor performance)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)
* Soft Actor Critic with adjustable TD step size (tdn sac, this is a failed attempt)

## Under Development Algorithms (Unrunnable)
* [Model-based Policy Optimization](https://arxiv.org/abs/1906.08253) (MBPO)
* [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971v6) (DDPG)

---
## Quick Start
``` shell
conda env create -f env.yaml 
conda activate rl_base
source prep
cd "algorithm"
python main.py config_path(config/Hopper.json)  
```

