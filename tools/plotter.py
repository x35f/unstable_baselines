import seaborn as sns
import matplotlib.pyplot as plt
import click
import os
from plot_helper import create_log_pdframe, load_logs
import matplotlib.pyplot as plt
from tqdm import tqdm

ALGOS = [
    'sac',
    'td3',
]
TASKS = {
    'Humanoid-v3': 3000000,
    'Walker2d-v3': 3000000
}

KEY_MAPPING = {
    'performance/eval_return':'eval_return',
    #'performance/train_return':'train_return'
}

@click.command()
@click.argument("log-dir", type = str)
@click.option("--plot_interval", type = int, default = 10)
@click.option("--output_dir", type = str, default = "results/")
def main(log_dir, plot_interval, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #assert group_by in ['env', 'info']
    logs = load_logs(log_dir, algos=ALGOS, tasks=TASKS, keys=KEY_MAPPING.keys(), plot_interval=plot_interval) 
    df, value_keys = create_log_pdframe(logs, KEY_MAPPING)
    for task_name in TASKS:
        task_output_dir = os.path.join(output_dir, task_name)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        task_df = df.loc[df['task_name']==task_name].dropna()
        for value_key in tqdm(value_keys):
            value_df = task_df[['step', 'algo_name', value_key]]
            sns.lineplot(data = value_df,x = 'step', y = value_key, hue = 'algo_name')
            value_key = value_key.replace("/", "-")
            output_path = os.path.join(task_output_dir, value_key + ".pdf")
            plt.savefig(output_path)
            plt.clf()


if __name__ == "__main__":
    main()