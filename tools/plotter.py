import seaborn as sns
import matplotlib.pyplot as plt
import click
import os
from plot_helper import load_tb_logs, create_log_pdframe
import matplotlib.pyplot as plt
from tqdm import tqdm

@click.command()
@click.argument("path", type = str)
@click.option("--hue", type = str, default = None)
@click.option("--output_dir", type = str, default = "results/default")
def main(path, hue, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #assert group_by in ['env', 'info']
    logs = load_tb_logs(path) 
    pd_frame, value_keys = create_log_pdframe(logs)
    print("plotting")
    for value_key in tqdm(value_keys):
        #print(pd_frame.columns)
        #print(type(value_key))
        sns.lineplot(data = pd_frame, y = value_key, hue = hue ,dropna=True)
        value_key = value_key.replace("/", "-")
        output_path = os.path.join(output_dir, value_key + ".pdf")
        plt.savefig(output_path)
        plt.clf()


if __name__ == "__main__":
    main()