default_args = {
  "env_name": "",
  "env":{
    "frameskip": 4,
    "resolution": (84, 84),
    "nstack": 4,
    "noop_max": 30
  },
  "common":{
    "n": 1,
    "gamma": 0.99
  },
  "buffer":{
    "max_buffer_size": 200000
  },
  "agent":{
    "double": True,
    "dueling": True,
    "target_smoothing_tau": 0.005,
    "update_target_network_interval": 1,
    "q_network":{
      "network_params": [("conv2d", 16, 8, 4, 0), ("conv2d", 32, 4, 2, 0),("flatten",), ("mlp", 256), ("mlp", 256)],
      "optimizer_class": "Adam",
      "learning_rate":0.0001,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_epoch": 4000,
    "num_updates_per_epoch":5000,
    "num_env_steps_per_epoch": 5000,
    "epsilon":{
      "initial_val": 1.0,
      "target_val": 0.05,
      "start_timestep": 0.0,
      "end_timestep": 2000000,
      "schedule_type": "linear"
    },
    "batch_size": 32,
    "max_trajectory_length":5000,
    "eval_interval": 10000,
    "num_eval_trajectories": 5,
    "snapshot_interval": 100000,
    "warmup_timesteps": 10000,
    "save_video_demo_interval": -1,
    "log_interval": 1000,
  }
  
}
