overwrite_args = {
    "algos":{ 
        "sac": "baselines/sac",
    },
    "tasks":[
        "Hopper-v3",
    ],
    "seeds": [0, 2],
    #"log_dir":"/home/xf/unstable_baselines/logs",
    "log-dir":"/home/xf/unstable_baselines/tools/bulk_runner/test_logs",
    "overwrite_args": {
        "sac": {
            "Hopper-v3": {"trainer/max_env_steps": "3000"}
        },

    }
}