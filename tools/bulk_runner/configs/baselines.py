overwrite_args = {
    "algos":{   # algorithms to run, the value represents the relative path to algo directory
        "ddpg": "baselines/ddpg",
        "ppo": "baselines/ppo",
        "redq": "baselines/redq",
        "sac": "baselines/sac",
        "td3": "baselines/td3",
        "trpo": "baselines/trpo",
        "vpg": "baselines/vpg",
    },
    "tasks":[
        "Hopper-v4",
        "HalfCheetah-v4",
        "Ant-v4",
        "Humanoid-v4",
        "Swimmer-v4",
        "Walker2d-v4"
    ],
    "seeds": [0, 10],
    #"log_dir":"/home/xf/unstable_baselines/logs",
    "log_dir":"/home/xf/unstable_baselines/logs/",
    "overwrite_args": {
    }
}