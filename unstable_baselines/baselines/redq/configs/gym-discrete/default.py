default_args = {
  "env_name": "",
  "env":{

  },
  "buffer":{
    "max_buffer_size": 100000
  },
  "agent":{
    "gamma": 0.99,
    "reward_scale": 5.0,
    "update_target_network_interval": 1,
    "target_smoothing_tau": 0.005,
    "num_q_networks": 10,
    "num_q_samples": 2,
    "alpha": 0.2,
    "q_network":{
      "network_params": [("mlp", 64), ("mlp", 64)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "network_params": [("mlp", 64), ("mlp", 64)],
      "optimizer_class": "Adam",
      "deterministic": False,
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity",
      "reparameterize": True,
      "stablelize_log_prob": True,
    },
    "entropy":{
      "automatic_tuning": True,
      "learning_rate": 0.0003,
      "optimizer_class": "Adam"
    }
  },
  "trainer":{
    "max_env_steps":200000,
    "batch_size": 256,
    "max_trajectory_length":1000,
    "update_policy_interval": 20,
    "eval_interval": 2000,
    "num_eval_trajectories": 10,
    "save_video_demo_interval": -1,
    "warmup_timesteps": 1000,
    "snapshot_interval": 10000,
    "log_interval": 100,
    "utd": 20
  }
}
