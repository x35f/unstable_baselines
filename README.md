rl codebase of feng xu
xufeng@lamda.nju.edu.cn'

usage:

conda create -f env.yaml # create a conda environment named rl_base
conda activate rl_base
source prep
cd "algorithm"
python main.py config_path(config/Hopper.json)  
