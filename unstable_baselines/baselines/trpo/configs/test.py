overwrite_args = {
  "env_name": "Hopper-v3",
  "common":{
    "gamma": 0.99,
    "n": 1,
    "max_trajectory_length":100,
    "gae_lambda": 0.97
  },
  "buffer":{
    "num_trajs_per_iteration":-1,
    "max_steps_per_iteration": 20,
    "advantage_type": "gae"
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
    "train_pi_iters": 1,
    "train_v_iters": 1,
    "normalize_advantage": True,
    "v_network":{
      "hidden_dims": [20,20],
      "optimizer_class": "Adam",
      "learning_rate":0.001,
      "act_fn": "tanh",
      "out_act_fn": "tanh"
    },
    "policy_network":{
      "hidden_dims": [20, 20],
      "optimizer_class": "Adam",
      "deterministic": False,
      "learning_rate":0.0003,
      "act_fn": "tanh",
      "out_act_fn": "tanh",
      "re_parameterize": False
    }
  },
  "trainer":{
    "max_iteration": 3,
    "epoch": 2,
    "batch_size": 4,
    "test_interval": 1,
    "num_test_trajectories": 2,
    "snapshot_interval": 1,
    "start_timestep": 0,
    "save_video_demo_interval": 1,
    "log_interval": 1
  },
  "env":{
  }
  
}
