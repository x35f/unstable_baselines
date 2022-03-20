default_args = {
  "env_name": "",
  "buffer":{
    "gamma": 0.99,
    "size": 1000,
    "advantage_type": "gae",
    "gae_lambda": 0.97,
    "normalize_advantage": True,
    "max_trajectory_length": 1000
  },
  "agent": {
    "gamma": 0.99,
    "train_v_iters": 80,
    "action_bound_method": "clip",
    "v_network": {
      "hidden_dims": [64, 64],
      "optimizer_class": "Adam",
      "learning_rate": 1e-3,
      "act_fn": "tanh",
      "out_act_fn": "identity"
    },
    "policy_network": {
      "hidden_dims": [64, 64],
      "optimizer_class": "Adam",
      "learning_rate": 3e-4,
      "act_fn": "tanh",
      "out_act_fn": "identity",
      "re_parameterize": False,
      "fix_std": True
    }
  },
  "trainer": {
    "max_env_steps": 2000000,
    "max_trajectory_length": 1000,
    "num_env_steps_per_epoch": 1000,
    "num_eval_trajectories": 10,
    "log_interval": 2000,
    "eval_interval": 10000,
    "snapshot_interval": 100000,
    "save_video_demo_interval": 100000
  }
}

