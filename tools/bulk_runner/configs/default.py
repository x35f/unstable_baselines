default_args = {
    "gpu_ids": [0, 1], # cuda device ids
    "max_exps_per_gpu": 2, # max number of experiments to run on a single gpu
    "estimated_gpu_memory_per_exp": 3000,  # by megabytes
    "estimated_system_memory_per_exp": 10000,  # by megabytes
    "refresh_interval": 120, # seconds between refreshing gpu status 
    "algos":{   # algorithms to run, the value represents the relative path to the unstable_baselines installed direcotry
    },
    "tasks":[ # tasks to run

    ],
    "seeds": [0], 
    "log-dir":"",
    "overwrite_args": { # additional arguments to overwrite for each experiment
        # "sac": {
        #     "Hopper-v3": {"trainer/max_env_steps": "3000"},
        #     "Walker2d-v3": {"trainer/max_env_steps": "3000"}
        # },
        # "td3": {
        #     "Hopper-v3": {"trainer/max_env_steps": "3000"},
        #     "Walker2d-v3": {"trainer/max_env_steps": "3000"}
        # },

    }
}