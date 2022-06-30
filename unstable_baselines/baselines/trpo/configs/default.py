default_args = {
  "common":{
    "max_trajectory_length":1000
  },
  "buffer":{
    "size": 20000,
  },
  "agent":{
    "gamma": 0.995,
    "tau": 0.97,
    "l2_reg": 1e-3,
    "v_optimize_maxiter": 25,
    "num_conjugate_gradient_steps": 10,
    "damping": 0.1,
    "max_kl_div": 1e-2,
    "max_backtracks": 10,
    "accept_ratio": 0.1,
    "residual_tol": 1e-10,
    "v_network":{
      "hidden_dims": [64,64],
      "act_fn": "tanh",
      "out_act_fn": "identity"
    },
    "policy_network":{
      "hidden_dims": [64, 64],
      "deterministic": False,
      "act_fn": "tanh",
      "out_act_fn": "identity",
      "re_parameterize": False,
      "predicted_std": False,
      "paramterized_std": True,
      "stablize_log_prob": False
    }
  },
  "trainer":{
    "max_env_steps": 3000000,
    "num_env_steps_per_epoch": 15000,
    "eval_interval": 10,
    "num_eval_trajectories": 5,
    "snapshot_interval": 100,
    "start_timestep": 0,
    "save_video_demo_interval": -1,
    "log_interval": 1
  },
  "env":{
  }
  
}
