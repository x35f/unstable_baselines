overwrite_args = {
  "env_name": "CartPole-v1",
  "common":{
    "n": 1,
    "gamma": 0.99
  },
  "buffer":{
    "max_buffer_size": 100
  },
  "agent":{
    "update_target_network_interval": 1,
    "tau": 0.3,
    "q_network":{
      "hidden_dims": [12,12],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_epoch": 5,
    "num_updates_per_epoch":5,
    "num_env_steps_per_epoch": 5,
    "batch_size": 4,
    "max_trajectory_length": 100,
    "eval_interval": 1,
    "num_eval_trajectories": 2,
    "snapshot_interval": 1,
    "start_timestep": 1,
    "save_video_demo_interval": 1,
    "log_interval": 5,
    "epsilon": 0.2
  },
  "env":{
  }
  
}
