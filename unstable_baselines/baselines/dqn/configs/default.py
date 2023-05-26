default_args = {
  "env_name": "",
  "common":{
    "n": 1,
    "gamma": 0.99
  },
  "buffer":{
    "max_buffer_size": 1000000
  },
  "agent":{
    "double": True,
    "dueling": True,
    "target_smoothing_tau": 1.0,
    "update_target_network_interval": 50,
    "q_network":{
      "network_params": [("mlp", 32), ("mlp", 16)],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_epoch": 2000,
    "num_env_steps_per_epoch": 50,
    "num_updates_per_epoch":50,
    "epsilon":{
      "initial_val": 1.0,
      "target_val": 0.1,
      "start_timestep": 0.0,
      "end_timestep": 10000,
      "schedule_type": "linear"
    },
    "batch_size": 64,
    "max_trajectory_length":500,
    "eval_interval": 2000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 5000,
    "warmup_timesteps": 1000,
    "save_video_demo_interval": -1,
    "log_interval": 200
  }
  
}
