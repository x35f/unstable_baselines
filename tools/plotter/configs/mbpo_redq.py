overwrite_args = {
    "mode": "joint",
    "algos": [     
        'mbpo',
        'redq'
        ],
    "tasks":{
        'Ant-v3': 500000,
        'HalfCheetah-v3': 500000,
        'Hopper-v3': 500000,
        'Humanoid-v3': 500000,
        "Swimmer-v3": 500000,
        'Walker2d-v3': 500000,
    },
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (3,3),
    "output_dir": "results/mbpo_redq"
}