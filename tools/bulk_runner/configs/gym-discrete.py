overwrite_args = {
    "algos":{   # algorithms to run, the value represents the relative path to algo directory
        "ppo": "baselines/ppo",
        "sac": "baselines/sac",
        "redq": "baselines/redq",
    },
    "tasks":[
        "gym-discrete/Acrobot-v1",
        "gym-discrete/CartPole-v1",
        "gym-discrete/LunarLander-v2",
        "gym-discrete/MountainCar-v0",
    ],
    "seeds": [0, 10],
    #"log_dir":"/home/xf/unstable_baselines/logs",
    "log-dir":"./algo_logs/",
    "overwrite_args": {
    }
}