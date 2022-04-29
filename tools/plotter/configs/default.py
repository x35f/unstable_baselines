default_args = {
    "algos": [ #List: Name of algorithms to plot
        ],
    "tasks":{ # Dict, env_name: max timestep. Name of tasks to plot
    },
    'key_mapping':{ # Dict, tb key name: name to appear on figure.
    },
    "aspect": 1.2, # length/height ratio for each figure/subfigure
    "mode": "joint",
    "col_wrap": 3, 
    "plot_interval": 10,
    "smooth_length": 0,
    "x_axis_sci_limit": (0,0)

}