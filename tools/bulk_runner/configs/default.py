default_args = {
    "gpu_ids": [0, 1], # cuda device ids
<<<<<<< HEAD
    "max_exps_per_gpu": 2, # max number of experiments to run on a single gpu
    "estimated_gpu_memory_per_exp": 2000,  # by megabytes
    "estimated_system_memory_per_exp": 20000,  # by megabytes
    "algos":{   # algorithms to run, the value represents the relative path to algo directory
        "sac": "../../unstable_baslines/baselines/sac",
        "ppo": "../../unstable_baslines/baselines/ppo"
    },
    "tasks":{
        "Hopper-v3",
        "HalfCheetah-v3"
    },
    "seeds": [0, 10, 20, 30],
    "log_dir":"/home/xf/unstable_baselines/logs",
    "overwrite_args": {
        "sac": {
            "Hopper-v3": {"trainer/max_env_steps": "3000"}
=======
    "max_exps_per_gpu": 3, # max number of experiments to run on a single gpu
    "estimated_gpu_memory_per_exp": 2000,  # by megabytes
    "estimated_system_memory_per_exp": 10000,  # by megabytes
    "refresh_interval": 30, # seconds between refreshing gpu status 
    "algos":{   # algorithms to run, the value represents the relative path to the unstable_baselines installed direcotry
        "sac": "baselines/sac",
        "td3": "baselines/td3",
    },
    "tasks":[ # tasks to run
        "Hopper-v3",
        "Walker2d-v3"
    ],
    "seeds": [0, 10, 20], 
    "log-dir":"/home/xf/unstable_baselines/logs",
    "overwrite_args": { # additional arguments to overwrite for each experiment
        "sac": {
            "Hopper-v3": {"trainer/max_env_steps": "3000"},
            "Walker2d-v3": {"trainer/max_env_steps": "3000"}
        },
        "td3": {
            "Hopper-v3": {"trainer/max_env_steps": "3000"},
            "Walker2d-v3": {"trainer/max_env_steps": "3000"}
>>>>>>> mbpo_env_update
        },

    }
}