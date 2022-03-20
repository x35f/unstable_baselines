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
    "max_env_steps": 1000,
    "batch_size": 16,
    "max_trajectory_length":200,
    "test_interval": 1,
    "num_test_trajectories": 2,
    "snapshot_interval": 1,
    "start_timestep": 1,
    "random_policy_timestep": 1,
    "save_video_demo_interval": 1,
    "log_interval": 1
  }
}
