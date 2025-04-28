import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re

path = "/cs/student/projects1/2021/rstewart/code/training_history/"
folders = ["bull1_1_length", "bull1_3_length", "bull2_1_length", "bull2_3_length", "love_parade_1_length", "love_parade_3_length"]
name_map = {"fno":"FNO", "localfno":"LNO", "spatiotemporalfno":"STFNO", "tfno":"TFNO", "uno":"UNO"}
model_list = ["FNO", "LNO", "STFNO", "TFNO", "UNO"]
bright_colors = [
    '#e6194b',  # red
    '#3cb44b',  # green
    '#ffe119',  # yellow
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#46f0f0',  # cyan
    '#f032e6',  # pink
    '#bcf60c',  # lime
    '#fabebe'   # light pink
]

for folder in folders:
    folder_path = os.path.join(path, folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        match = re.search(r'training_history_(.*?)_', filename)
        if match:
            model_name = match.group(1)
        model_name = name_map.get(model_name)

        model_color_map = {
            model: bright_colors[i % len(bright_colors)]
            for i, model in enumerate(sorted(model_list))
        }

        plt.plot(df['Epoch'], df['MSE Loss'], label=model_name, color=model_color_map.get(model_name, 'black'))
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Training History Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder + '.png', dpi=300)
    plt.clf()   
    plt.close() 


print("Finished plotting training histories.")


