from cv2 import COLORMAP_WINTER
import seaborn as sns
import matplotlib.pyplot as plt
import click
import os
from plot_helper import create_log_pdframe, load_logs
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set_theme(style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True, rc=None)
#sns.set_style('darkgrid', {'legend.frameon':True})

# algorithms to plot, the order also stays the same in the legend
ALGOS = [
    'sac',
    'td3',
    'ddpg',
    'ppo',
    'vpg'
    # 'mbpo'
]
TASKS = {
    'Ant-v3': 3000000,
    'HalfCheetah-v3': 3000000,
    'Hopper-v3': 3000000,
    'Humanoid-v3': 3000000,
    "Swimmer-v3": 3000000,
    'Walker2d-v3': 3000000,
}
# TASKS = {
# 'AntTruncatedObs-v2': 300000,
# 'HalfCheetah-v2': 400000,
# 'Hopper-v2': 125000,
# 'HumanoidTruncatedObs-v2': 300000,
# 'InvertedPendulum-v2': 15000,
# 'Walker2d-v2': 300000
# }


KEY_MAPPING = {
    'performance/eval_return':'Eval Return',
    'performance/train_return':'Train Return'
}

#parameters for joint plot
COL_WRAP = 3 # number of subfigures for each row
ASPECT = 1.2 # length/height ratio for each subplot


def single_plot(df, value_keys, output_dir):
    for task_name in tqdm(TASKS):
        task_output_dir = os.path.join(output_dir, task_name)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        task_df = df.loc[df['task_name']==task_name]
        for value_key in value_keys:
            value_df = task_df[['timestep', 'algo_name', value_key]].dropna()
            sns.lineplot(data = value_df,x = 'timestep', y = value_key, hue = 'algo_name')
            output_path = os.path.join(task_output_dir, value_key + ".pdf")
            plt.savefig(output_path)
            plt.clf()

def joint_plot(df, value_keys, output_dir):
    global ALGOS, COL_WRAP, ASPECT
    for value_key in tqdm(value_keys):
        value_df = df[['timestep', 'algo_name', value_key,'task_name']]
        g = sns.FacetGrid(value_df, col='task_name',hue="algo_name", hue_order=ALGOS, sharex=False, sharey=False, col_wrap=COL_WRAP, legend_out=True, aspect=ASPECT)
        g.map(sns.lineplot, "timestep", value_key,)
        g.add_legend(loc="center right",frameon=True, title="", ncol=1)
        g.set_titles(col_template="{col_name}")#, row_template="{row_name}")
        axes = g.axes.flatten()
        #axes[0].set_ylabel("Number of Defects")
        for ax in axes:
            ax.set_xlabel("time step")
        #g.tight_layout()
        output_path = os.path.join(output_dir, value_key + '.pdf')
        g.savefig(output_path)
        plt.clf()

@click.command()
@click.argument("log-dir", type = str)
@click.option("--plot_interval", type = int, default = 10)
@click.option("--smooth_length", type = int, default = 0)
@click.option("--mode", type = str, default = "single")
@click.option("--output_dir", type = str, default = "results/")
def main(log_dir, plot_interval, smooth_length, mode, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #assert group_by in ['env', 'info']
    print("loading logs")
    logs = load_logs(log_dir, algos=ALGOS, tasks=TASKS, keys=KEY_MAPPING.keys(), plot_interval=plot_interval, smooth_length=smooth_length) 
    df, value_keys = create_log_pdframe(logs, KEY_MAPPING)

    print("plotting")
    if mode == "single": # one figure for each task
        single_plot(df, value_keys, output_dir)
    elif mode == "joint": # plot tasks together in one figure
        joint_plot(df, value_keys, output_dir)

    



if __name__ == "__main__":
    main()