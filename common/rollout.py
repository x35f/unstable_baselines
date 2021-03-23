from common.buffer import ReplayBuffer, TDReplayBuffer

def rollout(env, agent, max_env_steps, max_trajectories=-1, max_traj_length=1000, n=1, gamma=0.99):
    #max_env_steps: max environments to sample
    #max_trajectories: max trajectories to sample
    #max_traj_length: max length of each trajectory
    #n: for td learning
    max_rollout_buffer_size = max_env_steps + max_traj_length # in case an additional full trajectory is sampled
    if n == 1:
        #naive buffer case
        rollout_buffer = ReplayBuffer(env.observation_space, env.action_space, max_buffer_size=max_rollout_buffer_size)
    else:
        rollout_buffer = TDReplayBuffer(env.observation_space, env.action_space, n=n, gamma=gamma, max_buffer_size=max_rollout_buffer_size)
    for i in range(max_trajectories):
        states, actions, next_states, rewards, dones = rollout_trajectory(env, agent, max_traj_length)
        rollout_buffer.add_traj(states, actions, next_states, rewards, dones)
    return rollout_buffer


def rollout_trajectory(env, agent, max_traj_length):
    states, actions, next_states, rewards, dones = [], [], [], [], []
    state = env.reset()
    done = False
    traj_length = 0
    while not done:
        action = agent.sample(state)
        next_state, reward, done, info = env.step(action)
        traj_length += 1
        if traj_length >= max_traj_length:
            done = 1.
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        if done:
            break
    return states, actions, next_states, rewards, dones