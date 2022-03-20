overwrite_args = {
  "env_name": "Hopper-v3",
  "buffer":{
    "max_buffer_size": 1000
  },
  "agent":{
    "gamma": 0.99,
    "update_target_network_interval": 1,
    "target_smoothing_tau": 0.005,
    "q_network":{
      "hidden_dims": [20, 20],
      "optimizer_class": "Adam",
      "learning_rate":0.001,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "hidden_dims": [20, 20],
      "optimizer_class": "Adam",
      "deterministic": True,
      "learning_rate":0.001,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_env_steps": 10,
    "batch_size": 2,
    "max_trajectory_length":100,
    "eval_interval": 1,
    "num_eval_trajectories": 1,
    "snapshot_interval": 2,
    "start_timestep": 1,
    "save_video_demo_interval": 1,
    "log_interval": 1,
    "sequential": False,
    "action_noise_scale": 0.1
  },
  "env":{
    "reward_scale": 10.0
  }
  
}
