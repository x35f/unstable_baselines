# Reinforcement Learning Codebase of LAMDA5-Z

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

This is a LXH-unfriendly, but YGY-friendly and GCX-CrazyHappy project. Unstable Baselines is designed to provide a quick-start guide for Reinforcement Learning beginners and a codebase for agile algorithm development. In light of this, only the basic version of each algorithm is implemented here, without tedious training skills and code-level optimizations. 
UB is currently maintained by researchers from [lamda-rl](https://github.com/LAMDA-RL), and a pypi source will be available once it is ready for publishing.


---
## Stable algorithims (Runnable and have good performance):
* [Deep Q Learning](https://arxiv.org/abs/1312.5602) (DQN) 
* [Policy Gradient](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) (VPG)
* [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971v6) (DDPG)
* [Soft Actor Critic](https://arxiv.org/abs/1801.01290) (SAC)
* [Twin Delayed Deep Deterministic policy gradient algorithm](https://arxiv.org/pdf/1802.09477) (TD3)


## Unstable Algorithms (Runnable but does not have SOTA performance)
* [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982) (REDQ)
* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO)
* [Model-based Policy Optimization](https://arxiv.org/abs/1906.08253) (MBPO)
* [Efficient Off-policy Meta-learning via Probabilistic Context Variables](http://arxiv.org/abs/1903.08254) (PEARL)

## Under Development Algorithms (Unrunnable)

---
## Quick Start

### Install
``` shell
git clone --recurse-submodules https://github.com/x35f/unstable_baselines.git
cd unstable_baselines
conda env create -f env.yaml 
conda activate rl_base
pip3 install -e .
```

### To run an algorithm
``` shell
python3 /path/to/algorithm/main.py /path/to/algorithm/configs/some-config.json args(optional)
```

### Install environments (optional)
``` shell
#install metaworld for meta_rl benchmark
cd envs/metaworld
pip install -e .
#install atari
pip install gym[all]
```

