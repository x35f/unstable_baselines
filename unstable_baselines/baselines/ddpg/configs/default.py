default_args = {
  "env_name": "",
  "buffer":{
    "max_buffer_size": 1000000
  },
  "agent":{
    "gamma": 0.99,
    "target_smoothing_tau": 0.005,
    "q_network":{
      "hidden_dims": [256,256],
      "optimizer_class": "Adam",
      "learning_rate":0.001,
      "act_fn": "relu",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "hidden_dims": [256,256],
      "optimizer_class": "Adam",
      "deterministic": True,
      "learning_rate":0.001,
      "act_fn": "tanh",
      "out_act_fn": "identity"
    }
  },
  "trainer":{
    "max_env_steps": 3000000,
    "batch_size": 100,
    "max_trajectory_length":1000,
    "eval_interval": 2000,
    "num_eval_trajectories": 10,
    "snapshot_interval": 100000,
    "random_sample_timestep": 10000,
    "start_update_timestep": 2000,
    "update_interval": 50,
    "save_video_demo_interval": 50000,
    "log_interval": 1000,
    "action_noise_scale": 0.1
  }
  
}
