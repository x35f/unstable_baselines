default_args = {
  "env_name": "",
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
      "hidden_dims": [256,256],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "hidden_dims": [256,256],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity",
      "re_parameterize": True, 
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
    "max_env_steps": 3000000,
    "batch_size": 256,
    "max_trajectory_length":1000,
    "eval_interval": 10000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 200000,
    "start_timestep": 2000,
    "random_policy_timestep": 5000, 
    "save_video_demo_interval": -1,
    "log_interval": 1000
  }
}
