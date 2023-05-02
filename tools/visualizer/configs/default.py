default_args = {
    'width': 256, # Int: video wdith
    'height': 256,  # Int: video height
    'fps': 30, # Int: frame per second
    'num_trials': 5, # Int: number of trials to run to select the best trial
    'mode':"best",  # Str[best/last]: "best" automatically selects the best model from all saved models, "last" loads the last saved model
    'output_dir': 'results', # Str: directory to save the output video
    'max_trajectory_length': 1000   # Int: max trajectory length
}