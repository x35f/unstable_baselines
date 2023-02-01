overwrite_args = {
    "algos":{   # algorithms to run, the value represents the relative path to algo directory
        "sac": "../../unstable_baselines/baselines/sac",
        "ppo": "../../unstable_baselines/baselines/ppo"
    },
    "tasks":[
        "Hopper-v3",
        "HalfCheetah-v3"
    ],
    "seeds": [0, ],
    #"log_dir":"/home/xf/unstable_baselines/logs",
    "log_dir":"/home/xf/unstable_baselines/tools/bulk_runner/test_logs",
    "overwrite_args": {
        "sac": {
            "Hopper-v3": {"trainer/max_env_steps": "3000"}
        },
        "ppo": {
            "HalfCheetah-v3": {"trainer/max_env_steps": "3000"}
        },

    }
}