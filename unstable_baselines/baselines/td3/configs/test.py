overwrite_args = {
  "env_name": "Hopper-v3",
  "buffer":{
    "max_buffer_size": 1000
  },
  "agent":{
    "q_network":{
      "hidden_dims": [16,16]
    },
    "policy_network":{
      "hidden_dims": [16,16]
    }
  },
  "trainer":{
    "policy_delay": 2,
    "max_env_steps": 10,
    "batch_size": 4,
    "max_trajectory_length":10,
    "eval_interval": 11,
    "num_eval_trajectories": 2,
    "snapshot_interval": 1,
    "random_sample_timestep": 1,
    "start_update_timestep": 1,
    "update_interval": 1,
    "save_video_demo_interval": 1,
    "log_interval": 1
  }
}
