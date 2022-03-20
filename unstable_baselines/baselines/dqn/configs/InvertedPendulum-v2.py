overwrite_args = {
  "env_name": "InvertedPendulum-v2",
  "agent":{
    "update_target_network_interval": 1,
    "tau": 0.05,
    "q_network":{
      "hidden_dims": [64,64],
      "optimizer_class": "Adam",
      "learning_rate":0.0003,
      "act_fn": "relu",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_iteration": 2000000,
    "num_updates_per_iteration":50,
    "num_steps_per_iteration": 50,
    "batch_size": 64,
    "max_trajectory_length":200,
    "test_interval": 100,
    "num_test_trajectories": 3,
    "snapshot_interval": 100,
    "start_timestep": 1000,
    "save_video_demo_interval": 100,
    "log_interval": 5,
    "epsilon": 0.2
  }
}
