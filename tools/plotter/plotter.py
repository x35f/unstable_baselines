from cv2 import COLORMAP_WINTER
import seaborn as sns
import matplotlib.pyplot as plt
import click
import os
from plot_helper import create_log_pdframe, load_logs
from unstable_baselines.common.util import load_config
import matplotlib.pyplot as plt
from tqdm import tqdm

sns.set_theme(style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True, rc=None)

def single_plot(df, value_keys, config, output_dir):
    tasks = config['tasks']
    x_axis_sci_limit = config['x_axis_sci_limit']
    for task_name in tqdm(tasks):
        task_output_dir = os.path.join(output_dir, task_name)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        task_df = df.loc[df['task_name']==task_name]
        for value_key in value_keys:
            value_df = task_df[['timestep', 'algo_name', value_key]].dropna()
            sns.lineplot(data = value_df,x = 'timestep', y = value_key, hue = 'algo_name')
            
            plt.ticklabel_format(axis="x", style="sci", scilimits=x_axis_sci_limit)
            output_path = os.path.join(task_output_dir, value_key + ".svg")
            plt.savefig(output_path)
            plt.clf()

def joint_plot(df, value_keys, config, output_dir):
    algos = config['algos']
    col_wrap = config['col_wrap']
    aspect = config['aspect']
    x_axis_sci_limit = config['x_axis_sci_limit']
    for value_key in tqdm(value_keys):
        value_df = df[['timestep', 'algo_name', value_key,'task_name']]
        g = sns.FacetGrid(value_df, col='task_name',hue="algo_name", hue_order=algos, sharex=False, sharey=False, col_wrap=col_wrap, legend_out=True, aspect=aspect)
        g.map(sns.lineplot, "timestep", value_key,)
        g.add_legend(loc="center right",frameon=True, title="", ncol=1)
        g.set_titles(col_template="{col_name}")#, row_template="{row_name}")
        axes = g.axes.flatten()
        for ax in axes:
            ax.set_xlabel("time step",fontsize="small" )
            ax.ticklabel_format(axis="x", style="sci", scilimits=x_axis_sci_limit)
        output_path = os.path.join(output_dir, value_key + '.svg')
        g.savefig(output_path)
        plt.clf()

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,)
)
@click.argument("config-path", type = str) # path to plot config
@click.argument("log-dir", type = str)  # path to log dir, the logs follows the default format of usb
@click.argument('args', nargs=-1)   # args in the config to overwrite 
def main(config_path, log_dir, args):
    # load config
    config = load_config(config_path, args)
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # load logs
    print("loading logs")
    logs = load_logs(log_dir, config) 
    df, value_keys = create_log_pdframe(logs, config['key_mapping'])

    # plot 
    print("plotting")
    mode = config['mode']
    if mode == "single": # one figure for each task
        single_plot(df, value_keys, config, output_dir)
    elif mode == "joint": # plot tasks together in one figure
        joint_plot(df, value_keys, config, output_dir)

if __name__ == "__main__":
    main()