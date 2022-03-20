default_args = {
  "common":{
    "max_trajectory_length":1000
  },
  "buffer":{
    "advantage_type": "gae",
    "size": 4000,
    "gamma": 0.99,
    "normalize_advantage":true,
    "gae_lambda": 0.97
  },
  "agent":{
    "beta": 1.0,
    "policy_loss_type": "clipped_surrogate",
    "entropy_coeff": 0.0,  
    "c1": 1.0,
    "c2": 1.0,
    "clip_range": 0.2,
    "target_kl": 0.01,
    "adaptive_kl_coeff": False,
    "train_policy_iters": 80,
    "train_v_iters": 80,
    "v_network":{
      "hidden_dims": [64,64],
      "optimizer_class": "Adam",
      "learning_rate":0.001,
      "act_fn": "tanh",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "hidden_dims": [64, 64],
      "optimizer_class": "Adam",
      "deterministic": False,
      "learning_rate":0.0003,
      "act_fn": "tanh",
      "out_act_fn": "identity",
      "re_parameterize": False,
      "fix_std": True,
      "paramterized_std": True,
      "stablize_log_prob": False
    }
  },
  "trainer":{
    "max_env_steps": 3000000,
    "num_env_steps_per_epoch": 4000,
    "batch_size": 64,
    "eval_interval": 10,
    "num_eval_trajectories": 5,
    "snapshot_interval": 100,
    "start_timestep": 0,
    "save_video_demo_interval": 300,
    "log_interval": 1
  },
  "env":{
  }
  
}
