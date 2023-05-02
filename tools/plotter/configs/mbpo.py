overwrite_args = {
    "mode": "joint",
    "algos": [
            'mbpo',
        ],
    "tasks":{
        'InvertedPendulum-v2': 15000,
        'Hopper-v2': 125000,
        'Walker2d-v2': 300000,
        'AntTruncatedObs-v2': 300000,
        'HalfCheetah-v2': 400000,
        'HumanoidTruncatedObs-v2': 300000
    },
    "plot_interval": 1,
    "smooth_length": 0,
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (3,3),
    "output_dir": "results/mbpo"
}