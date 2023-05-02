overwrite_args = {
    
    "refresh_interval": 3, # seconds between refreshing gpu status 
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
        "test"
    ],
    "seeds": [0, ],
    #"log_dir":"/home/xf/unstable_baselines/logs",
    "log_dir":"/home/xf/unstable_baselines/tools/bulk_runner/logs/",
    "overwrite_args": {
    }
}