overwrite_args = {
  "env_name": "",
  "common":{
    "n": 1,
    "gamma": 0.99
  },
  "buffer":{
    "max_buffer_size": 1000000
  },
  "agent":{
    "target_smoothing_tau": 0.3,
    "q_network":{
      "hidden_dims": [12,12],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_epoch": 2000,
    "num_updates_per_epoch":50,
    "num_env_steps_per_epoch": 50,
    "batch_size": 64,
    "max_trajectory_length":500,
    "eval_interval": 1000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 200,
    "start_timestep": 1000,
    "save_video_demo_interval": -1,
    "log_interval": 5,
    "epsilon": 0.2
  }
  
}
