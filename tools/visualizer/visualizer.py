import cv2
import json
import click
import os
import importlib
from tensorboard.backend.event_processing import event_accumulator
import torch
import numpy as np
from unstable_baselines.common.env_wrapper import get_env
from tqdm import tqdm


def load_config(log_dir):
    config_path = os.path.join(log_dir, 'parameters.txt')
    with open(config_path, 'r') as f:
        params = json.load(f)
    return params


def evaluate_agent(agent, env, num_trials=5):
    returns = []
    for trial_id in range(num_trials):
        obs = env.reset()
        for i in range()

    
def rollout(agent, env, width, height, max_trajectory_length, ret_imgs):
    imgs = []
    traj_ret = 0
    obs = env.reset()
    if ret_imgs:
        img = env.render(mode='rgb_array', width=width, height=height)
        imgs.append(img)
    for step in range(max_trajectory_length):
        action = agent.select_action(obs)['action']
        next_obs, reward, done, _ = env.step(action)
        traj_ret += reward
        obs = next_obs
        if ret_imgs:
            img = env.render(mode='rgb_array', width=width, height=height)
        if done:
            break
    return {
        "ret": traj_ret,
        "imgs": imgs if ret_imgs else None
    }
    

def select_best_snapshot(agent, env, snapshot_dirs, config, num_trials=5):
    best_snapshot_dir = ""
    best_ret = -np.inf
    print("selecting best snapshot")
    for snapshot_dir in tqdm(snapshot_dirs):
        for network_name, net in agent.networks.items():
            load_path = os.path.join(snapshot_dir, network_name + ".pt")
            agent.__dict__[network_name] = torch.load(load_path)
        rets = []
        for trial in range(num_trials):
            traj_ret = rollout(agent, env, ret_imgs=False, **config)
            rets.append(traj_ret)
        ret_mean = np.mean(rets)
        if ret_mean > best_ret:
            best_snapshot_dir = snapshot_dir
            best_ret = ret_mean
    return best_ret


def load_snapshot(agent, log_dir, mode):
    #get model path
    snapshot_dir = os.path.join(log_dir, "models")
    snapshot_dirs = [d for d in os.listdir(snapshot_dir) if "ite_" in d]
    snapshot_relative_dirs = [os.path.join(snapshot_dir, d) for d in snapshot_dirs]
    snapshot_timestamps = [int(d[4:]) for d in snapshot_dirs]
    snapshot_timestamps = sorted(snapshot_timestamps)
    if mode == 'last':
        selected_timestamp = snapshot_timestamps[-1]
    elif mode == 'best':
        selected_dir = select_best_snapshot(agent, snapshot_relative_dirs)
        selected_timestamp = int(selected_dir.split(os.sep)[-1][4:])

    selected_snapshot_dir = os.path.join(snapshot_dir, "ite_"+str(selected_timestamp))
    for network_name in agent.networks.items():
        load_path = os.path.join(selected_snapshot_dir, network_name + ".pt")
        agent.__dict__[network_name] = torch.load(load_path)

@click.command()
@click.argument("algo_dir", type=str)
@click.argument("log_dir", type=str)
@click.argument("config-path", type=str)
@click.option("--num_trials", type=int, default=5)
@click.option("--mode", type=str, default='last')
@click.option("output_path", type=str, default="videos/")
def main(algo_dir, log_dir, config_path, num_trials, output_path):
    algo_name = algo_dir.split(os.sep)[-1]
    #load config
    params = load_config(log_dir)

    #load env
    env_name = params['env_name']
    env = get_env(env_name)
    obs_space = env.observation_space
    action_space = env.action_space
    max_trajectory_length = params['trainer']['max_trajectory_length']

    #load agent
    agent_path = os.path.join(algo_dir,'agent.py')
    agent_name = algo_name.upper + "Agent"
    agent_module_str = agent_path.replace(".py","").replace(os.path.sep, '.')
    agent_module = importlib.importmodule(agent_module_str)
    agent_class = getattr(agent_module, agent_name)
    agent = agent_class(obs_space, action_space,**params['agent'])

    #load model
    load_snapshot(agent, log_dir, max_trajectory_length)

    #save video demo
    for trail_id in range(num_trials):


if __name__ == "__main__":
    main()