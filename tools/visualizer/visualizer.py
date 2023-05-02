import cv2
import json
import click
import os
import importlib
from tensorboard.backend.event_processing import event_accumulator
import torch
import numpy as np
from unstable_baselines.common.env_wrapper import get_env
from unstable_baselines.common.util import load_config, set_device_and_logger
from tqdm import tqdm
from operator import itemgetter
from unstable_baselines.common import util

AGENT_MODULE_MAPPING={
    "sac":"unstable_baselines.baselines.sac.agent",
    "ddpg":"unstable_baselines.baselines.ddpg.agent",
    "dqn":"unstable_baselines.baselines.dqn.agent",
    "ppo":"unstable_baselines.baselines.ppo.agent",
    "redq":"unstable_baselines.baselines.redq.agent",
    "td3":"unstable_baselines.baselines.td3.agent",
    "vpg":"unstable_baselines.baselines.vpg.agent",
    "mbpo":"unstable_baselines.model_based_rl.mbpo.agent",
    "pearl": "unstable_baselines.meta_rl.pearl.agent"
}

def load_params(log_dir):
    config_path = os.path.join(log_dir, 'parameters.txt')
    with open(config_path, 'r') as f:
        params = json.load(f)
    return params
    
def rollout(agent, env, width, height, max_trajectory_length, ret_imgs, **args):

    imgs = []
    traj_ret = 0
    obs = env.reset()
    if ret_imgs:
        img = env.render(mode='rgb_array', width=width, height=height)
        imgs.append(img)
    for step in range(max_trajectory_length):
        #obs = torch.FloatTensor(obs).to(util.device)
        action = agent.select_action(obs)['action']
        next_obs, reward, done, _ = env.step(action)
        traj_ret += reward
        obs = next_obs
        if ret_imgs:
            img = env.render(mode='rgb_array', width=width, height=height)
            imgs.append(img)
        if done:
            break
    return {
        "ret": traj_ret,
        "imgs": imgs if ret_imgs else None
    }
    

def select_best_snapshot(agent, env, snapshot_dirs, config):
    best_snapshot_dir = ""
    best_ret = -np.inf
    for snapshot_dir in tqdm(snapshot_dirs):
        for network_name, net in agent.networks.items():
            load_path = os.path.join(snapshot_dir, network_name + ".pt")
            agent.__dict__[network_name] = torch.load(load_path, map_location=util.device)
        rets = []
        for trial in range(config['num_trials']):
            traj_ret = rollout(agent, env, ret_imgs=False, **config)['ret']
            rets.append(traj_ret)
        ret_mean = np.mean(rets)
        if ret_mean > best_ret:
            best_snapshot_dir = snapshot_dir
            best_ret = ret_mean
    return best_ret, best_snapshot_dir


def load_snapshot(agent, env, log_dir, config):
    #get model path
    snapshot_dir = os.path.join(log_dir, "models")
    snapshot_dirs = [d for d in os.listdir(snapshot_dir) if "ite_" in d]
    snapshot_relative_dirs = [os.path.join(snapshot_dir, d) for d in snapshot_dirs]
    snapshot_timestamps = [int(d[4:]) for d in snapshot_dirs]
    snapshot_timestamps = sorted(snapshot_timestamps)
    if config['mode'] == 'last':
        selected_timestamp = snapshot_timestamps[-1]
    elif config['mode'] == 'best':
        best_ret, selected_dir = select_best_snapshot(agent, env, snapshot_relative_dirs, config)
        selected_timestamp = int(selected_dir.split(os.sep)[-1][4:])

    selected_snapshot_dir = os.path.join(snapshot_dir, "ite_"+str(selected_timestamp))
    for network_name, net in agent.networks.items():
        load_path = os.path.join(selected_snapshot_dir, network_name + ".pt")
        agent.__dict__[network_name] = torch.load(load_path ,map_location=util.device)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,)
)
@click.argument("algo_name", type=str) # Name of the algorithm, should be in the AGENT_MODULE_MAPPING global variable
@click.argument("log_dir", type=str) # Path of the log directory
@click.argument("config-path", type=str) # Config path
@click.option("--gpu", type=int, default=-1) # Device to load agent, -1 for cpu, >=0 for CUDA gpu
@click.argument('args', nargs=-1)
def main(algo_name, log_dir, config_path, gpu, args):
    #set device
    set_device_and_logger(gpu, None)

    #load config and parameters
    params = load_params(log_dir)
    config = load_config(config_path, args)

    #load env
    env_name = params['env_name']
    env = get_env(env_name)
    obs_space = env.observation_space
    action_space = env.action_space

    #load agent
    agent_name = algo_name.upper() + "Agent"
    agent_module = importlib.import_module(AGENT_MODULE_MAPPING[algo_name],package=algo_name+".agent")
    agent_class = getattr(agent_module, agent_name)
    agent = agent_class(obs_space, action_space, **params['agent'])
    
    #load model
    load_snapshot(agent, env, log_dir, config)

    #save video demo

    #select the best traj from trials
    traj_imgs = []
    num_trials = config['num_trials']
    best_ret = -10000000000
    for trial in range(num_trials):
        imgs, traj_ret = itemgetter("imgs","ret")(rollout(agent, env, ret_imgs=True, **config))
        if traj_ret > best_ret:
            traj_imgs = imgs
            best_ret = traj_ret
            
    # write imgs to video
    output_dir = config['output_dir']
    output_path = os.path.join(output_dir, "{}_{}_{}.mp4".format(algo_name, env_name, int(best_ret)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_size = (config['width'], config['height'])
    fps = config['fps']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)
    for img in traj_imgs:
        video_writer.write(img)
    video_writer.release()



if __name__ == "__main__":
    main()