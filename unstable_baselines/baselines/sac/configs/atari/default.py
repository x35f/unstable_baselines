default_args = {
  "env_name": "",
  "env":{
    "frameskip": 4,
    "resolution": (84, 84),
    "nstack": 4,
    "noop_max": 30
  },
  "buffer":{
    "max_buffer_size": 1000000
  },
  "agent":{
    "gamma": 0.99,
    "update_target_network_interval": 1,
    "target_smoothing_tau": 0.005,
    "alpha": 0.2,
    "reward_scale": 1.0,
    "q_network":{
      "network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 256)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 256)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity",
      "re_parameterize": True, 
      "stablize_log_prob": True,
      "log_var_min": -20, 
      "log_var_max": 2
    },
    "entropy":{
      "automatic_tuning": True,
      "learning_rate": 0.0003,
      "optimizer_class": "Adam"
    }
  },
  "trainer":{
    "max_env_steps": 20000000,
    "batch_size": 256,
    "eval_interval": 10000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 200000,
    "start_timestep": 2000,
    "random_policy_timestep": 5000, 
    "save_video_demo_interval": -1,
    "log_interval": 1000,
    "max_trajectory_length": 1000
  }
}
