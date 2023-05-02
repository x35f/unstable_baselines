<div align="center">
<img width="90%" height="auto" src="./docs/images/logo.svg">
</div>

---

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rlworkgroup/metaworld/blob/master/LICENSE)

Unstable Baselines(USB) is designed to serve as a quick-start guide for Reinforcement Learning beginners and a codebase for agile algorithm development. The algorithms strictly follows the original implementations, and the performance of Unstable Baselines matches those in the original implementations. USB is currently maintained by researchers from [lamda-rl](https://github.com/LAMDA-RL).

---
Features
1. **Novice-friendly**: USB is written in simple python codes. The RL training procedures are highly decoupled, waiting to be your first RL playground.  
2. **Stick to the original implementations**: USB is as a benchmark framework for RL, thus the re-implementations strictly follows the original implementations. Tricks to achieve a higher performance are not implemented.
3. **Customized Environments**: You can customized you own environment as long as it has Gym-like interfaces.
 
---

## Implementation Details
| Baseline RL  | Continuous Action Space | Discrete Action Space | Image Input | Status |
| --- | --- | --- | --- | --- |
| [DQN](https://arxiv.org/abs/1312.5602)       |     <font color="#dd0000">&#10005;</font><br />   |   <font color="#00dd00">&#10004;</font><br />     |   <font color="#00dd00">&#10004;</font><br />   | <font color="#00dd00">Stable</font><br />  |
| [VPG](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)    |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />   |   <font color="#00dd00">&#10004;</font><br />      | <font color="#00dd00">Stable</font><br /> |
| [DDPG](https://arxiv.org/abs/1509.02971v6)  |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />     |   <font color="#00dd00">&#10004;</font><br />    | <font color="#00dd00">Stable</font><br /> |
| [TD3](https://arxiv.org/pdf/1802.09477)  |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />     |   <font color="#00dd00">&#10004;</font><br />    | <font color="#00dd00">Stable</font><br /> |
| [TRPO](https://arxiv.org/abs/1502.05477) |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />     |   <font color="#00dd00">&#10004;</font><br />   | <font color="#00dd00">Stable</font><br /> |
| [PPO](https://arxiv.org/abs/1707.06347) |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#00dd00">&#10004;</font><br />      |   <font color="#00dd00">&#10004;</font><br />    | <font color="#00dd00">Stable</font><br /> |
| [SAC](https://arxiv.org/abs/1801.01290)|    <font color="#00dd00">&#10004;</font><br />    |  <font color="#00dd00">&#10004;</font><br />      |  <font color="#00dd00">&#10004;</font><br />    |  <font color="#00dd00">Stable</font><br /> |
| [REDQ](https://arxiv.org/abs/2101.05982)|    <font color="#00dd00">&#10004;</font><br />    |   <font color="#00dd00">&#10004;</font><br />     |   <font color="#00dd00">&#10004;</font><br />   | <font color="#00dd00">Stable</font><br /> |
| [Option Critic](https://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/12-Bacon-14858.pdf)|   -   |   -    |   -   | <font color="#778899">Developing</font><br /> |

| Model Based RL | Continuous Action Space | Discrete Action Space | Image Input | Status |
| --- | --- | --- | --- | --- |
| [MBPO](https://arxiv.org/abs/1906.08253) |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />     |  <font color="#00dd00">&#10004;</font><br />      |<font color="#dd0000">Updating</font><br /> |

| Meta RL | Continuous Action Space | Discrete Action Space | Image Input | Status |
| --- | --- | --- | --- | --- |
| [PEARL](http://arxiv.org/abs/1903.08254) |    <font color="#00dd00">&#10004;</font><br />    |   <font color="#dd0000">&#10005;</font><br />     |  <font color="#dd0000">&#10005;</font><br />     |<font color="#dd0000">Updating</font><br /> |
| [MAML](https://github.com/cbfinn/maml)|   -   |   -    |   -   | <font color="#778899">Developing</font><br /> |

\*Updating: the algorithm is being developed to adapt to the latest USB version, and will be "Stable" soon

\*Developing: the algorithm is being implemented, and will appear on the project soon


---
## Supported environment benchmarks
 * [Gym](https://www.gymlibrary.dev/) ("Classic Control" and "Box2D")
 * [MuJoCo](https://mujoco.org/)
 * [Atari](https://www.gymlibrary.dev/environments/atari/index.html)
 * [dm_control](https://github.com/deepmind/dm_control)
 * [metaworld](https://meta-world.github.io/)
---

## Performance 
### MuJoCo
<img width="100%" height="auto" src="./docs/images/Eval-Return.svg">

## Quick Start

### Install
``` shell
git clone --recurse-submodules https://github.com/x35f/unstable_baselines.git
cd unstable_baselines
conda env create -f env.yaml 
conda activate usb
pip install -e .
```

### To run an algorithm
In the directory of the algorithm
``` shell
python3 /path/to/algorithm/main.py /path/to/algorithm/configs/some-config.py args(optional)
```
For example

``` shell
cd unstable_baselines/baselines/sac
python3 main.py configs/Ant-v3.py --gpu 0
```
or for the ease of aggregating logs
``` shell
python3 unstable_baselines/baselines/sac/main.py unstable_baselines/baselines/sac/configs/Ant-v3.py --gpu 0
```

### Install environments (optional)
``` shell
#install metaworld for meta_rl benchmark
cd envs/metaworld
pip install -e .
```

## TODO List
* Add Documentation
