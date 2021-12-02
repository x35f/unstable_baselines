# Reinforcement Learning Codebase of [Feng Xu](mailto:xufeng@lamda.nju.edu.cn)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

Unastable Baselines is designed to provide a quick-start guide for Reinforcement Learning beginners and a codebase for agile algorithm development. In light of this, only the basic version of each algorithm is implemented here, without tedious training skills and code-level optimizations, such as multithread sampling and multi-GPU training. 
UB is currently maintained by researchers from [lamda-rl](https://github.com/LAMDA-RL), and a pypi source will be available once it is ready for publishing.


---
## Stable algorithims (Runnable and have good performance):
* [Deep Q Learning](https://arxiv.org/abs/1312.5602) (DQN) 
* [Soft Actor Critic](https://arxiv.org/abs/1801.01290) (SAC)
* [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982) (REDQ)


## Unstable Algorithms (Runnable but have poor performance)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)
* Soft Actor Critic with adjustable TD step size (tdn sac, this is a failed research attempt)
* [Model-based Policy Optimization](https://arxiv.org/abs/1906.08253) (MBPO)

## Under Development Algorithms (Unrunnable)
* [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971v6) (DDPG)
* [Efficient Off-policy Meta-learning via Probabilistic Context Variables](http://arxiv.org/abs/1903.08254) (PEARL)

---
## Quick Start
``` shell
git clone --recurse-submodules https://github.com/x35f/unstable_baselines.git
cd unstable_baselines
conda env create -f env.yaml 
conda activate rl_base

pip3 install -e .
python3 /path/to/algorithm/main.py /path/to/algorithm/configs/some-config.json
```

```install envs (optional)
#install metaworld for meta_rl benchmark
cd envs/metaworld
pip install -e .
#install atari
pip install gym[all]
```

