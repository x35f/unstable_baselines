overwrite_args = {
    "mode": "joint",
    "algos": [        
        'sac',
        'td3',
        'ddpg',
        'ppo',
        'vpg',
        'trpo'
        ],
    "tasks":{
        'Ant-v3': 3000000,
        'HalfCheetah-v3': 3000000,
        'Hopper-v3': 3000000,
        'Humanoid-v3': 3000000,
        "Swimmer-v3": 3000000,
        'Walker2d-v3': 3000000,
    },
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (6,6),
    "output_dir": "results/baselines"
}