overwrite_args = {
    "mode": "single",
    "algos": [
            'mbpo',
        ],
    "tasks":{
        'AntTruncatedObs-v2': 300000,
        'HalfCheetah-v2': 400000,
        'Hopper-v2': 125000,
        'HumanoidTruncatedObs-v2': 300000,
        'InvertedPendulum-v2': 15000,
        'Walker2d-v2': 300000
    },
    'key_mapping':{
        'performance/eval_return':'Eval Return',
        'performance/train_return':'Train Return'
    },
    "col_wrap":3, 
    "aspect": 1.2

}