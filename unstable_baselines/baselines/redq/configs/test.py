overwrite_args = {
  "env_name": "Hopper-v3",
  "buffer":{
    "max_buffer_size": 100
  },
  "agent":{
    "gamma": 0.99,
    "q_network":{
      "hidden_dims": [20,20]
    },
    "policy_network":{
      "hidden_dims": [20,20]
    }
  },
  "trainer":{
    "max_env_steps": 20,
    "batch_size": 2,
    "max_trajectory_length":5,
    "eval_interval": 1,
    "num_eval_trajectories": 10,
    "start_timestep": 1,
    "snapshot_interval": 1,
    "save_video_demo_interval": 1,
    "log_interval": 1
  }
}
