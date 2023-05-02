overwrite_args = {
    "mode": "joint",
    "algos": [
            'pearl',
        ],
    "tasks":{
        'ant-dir': 10000000,
        'ant-goal': 10000000,
        'cheetah-dir': 10000000,
        'cheetah-vel': 10000000,
        'humanoid-dir': 10000000,
        'walker-rand-params': 10000000
    },
    "plot_interval": 1,
    "smooth_length": 0,
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'performance/train_return':'Train Return'
    },
    "x_axis_sci_limit": (5,5),
    "output_dir": "results/pearl"
}