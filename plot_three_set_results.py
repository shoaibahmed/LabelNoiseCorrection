import os
import sys
from natsort import natsorted
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <Model directories> <Output file name>")
    exit()
model_dirs = sys.argv[1:-2]
assert len(model_dirs) >= 1
model_name_list = sys.argv[-2].split(";")
output_file = sys.argv[-1]
assert output_file.endswith(".png")
print("Input directories:", model_dirs)
print("Output file:", output_file)

assert len(model_dirs) == len(model_name_list)

all_results = {}
is_cifar100 = None
for i, main_model_dir in enumerate(model_dirs):
    if main_model_dir.startswith("#"):
        print("Ignoring commented line:", main_model_dir)
        continue
    current_model_dir = main_model_dir.split(";")
    print("Model directories with seeds:", current_model_dir)
    
    for model_dir in current_model_dir:
        assert os.path.exists(model_dir), model_dir
        print("Loading model directory:", model_dir)
        models = glob(os.path.join(model_dir, "**/last_epoch_*.pth"), recursive=True)
        print(models)
        
        results = {}
        for model in models:
            model_name = os.path.splitext(os.path.split(model)[-1])[0]
            model_name_parts = model_name.split("_")
            
            assert model_name_parts[5] == "valAcc"
            val_acc = float(model_name_parts[6])
            assert 0. <= val_acc <= 100.
            
            assert model_name_parts[7] == "noise"
            noise_level = float(model_name_parts[8])
            assert noise_level in [float(x) for x in range(0, 100, 1)]
            
            assert model_name_parts[9] == "bestValLoss"
            best_val_acc = float(model_name_parts[10])
            assert 0. <= best_val_acc <= 100.
            
            print(f"Noise level: {noise_level} / Val acc: {val_acc} / Best val acc: {best_val_acc}")
            results[noise_level] = (val_acc, best_val_acc)
        
        if is_cifar100 is None:
            is_cifar100 = 'cifar100' in model_dir.lower()
        
        dir_name = model_name_list[i]
        print(dir_name)
        if dir_name in all_results:
            for noise_level in results:
                all_results[dir_name][noise_level].append(results[noise_level])
        else:
            all_results[dir_name] = {}
            for noise_level in results:
                all_results[dir_name][noise_level] = [results[noise_level]]
print(all_results)

model_types = list(all_results.keys())
noise_levels = []
for model_type in model_types:
    noise_levels += list(all_results[model_type].keys())
noise_levels = natsorted(np.unique(noise_levels))
assert len(noise_levels) == 1
print("Noise levels:", noise_levels)

plot_end_acc = False
print("All model names:", model_types)

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
cm = plt.get_cmap('hsv')
NUM_COLORS = len(model_types)
if NUM_COLORS > 10:
    marker_colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
else:
    marker_colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple", "tab:olive", "tab:gray", "tab:pink", "tab:brown", "tab:cyan"]

if len(noise_levels) == 1:
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 10)
    
    for plot_best in [False] if plot_end_acc else [True]:
        noise_level = noise_levels[0]
        mean_acc_list = []
        model_name_list = []
        std_acc_list = []
        for idx in range(len(model_types)):
            accs = [all_results[model_types[idx]][noise_level][i][1 if plot_best else 0] for i in range(len(all_results[model_types[idx]][noise_level]))]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Model type: {model_types[idx]} / Noise level: {noise_level} / Acc list: {accs} / Mean: {mean_acc} / Std acc: {std_acc}")
            mean_acc_list.append(mean_acc)
            std_acc_list.append(std_acc)
            model_name_list.append(model_types[idx])
        
        mean_acc_list = np.array(mean_acc_list)
        std_acc_list = np.array(std_acc_list)
        
        # ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
        x_pos = np.arange(len(model_name_list))
        buffer_acc = 0.5
        line = plt.bar(x_pos, mean_acc_list, yerr=std_acc_list, linewidth=2., color=marker_colors[:len(mean_acc_list)], 
                       alpha=0.6, error_kw=dict())
        # min_idx, max_idx = np.argmin(mean_acc), np.argmax(mean_acc)
        # min_val = mean_acc_list[min_idx] - std_acc_list[min_idx] - buffer_acc
        # max_val = mean_acc_list[max_idx] + std_acc_list[max_idx] + buffer_acc
        min_val = np.min(mean_acc_list - std_acc_list - buffer_acc)
        max_val = np.max(mean_acc_list + std_acc_list + buffer_acc)
        
        plt.ylim(min_val, max_val)
        plt.xticks(x_pos)
        ax.set_xticklabels(model_name_list)
        plt.xticks(rotation=90)

else:
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    for idx in range(len(model_types)):
        for plot_best in [False] if plot_end_acc else [True]:
            model_results = all_results[model_types[idx]]
            mean_acc_list = []
            noise_list = []
            std_acc_list = []
            for noise_level in noise_levels:
                if noise_level not in model_results:
                    print(f"Warning: Noise level ({noise_level}) results not found for model: {model_types[idx]}")
                    continue
                assert isinstance(model_results[noise_level], list), model_results[noise_level]
                accs = [model_results[noise_level][i][1 if plot_best else 0] for i in range(len(model_results[noise_level]))]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                print(f"Model type: {model_types[idx]} / Noise level: {noise_level} / Acc list: {accs} / Mean: {mean_acc} / Std acc: {std_acc}")
                mean_acc_list.append(mean_acc)
                std_acc_list.append(std_acc)
                noise_list.append(noise_level)
            
            mean_acc_list = np.array(mean_acc_list)
            std_acc_list = np.array(std_acc_list)
            
            line = plt.plot(noise_list, mean_acc_list, linewidth=4., marker=marker_list[idx % len(marker_list)],
                            color=marker_colors[idx], alpha=0.6, markeredgecolor='k', markersize=9, label=f"{model_types[idx]}{' (Best)' if plot_best and plot_end_acc else ''}")
            plt.fill_between(noise_list, mean_acc_list - std_acc_list, mean_acc_list + std_acc_list, color=marker_colors[idx], alpha=0.25)
            line[0].set_color(marker_colors[idx])
            line[0].set_linestyle(line_styles[(1 if plot_best else 0) % len(line_styles)])

font_size = 16
plt.xlabel('Models' if len(noise_levels) == 1 else 'Noise level (%)', fontsize=font_size)
plt.ylabel('Accuracy (%)', fontsize=font_size)
if len(noise_levels) > 1:
    plt.legend(prop={'size': font_size})
plt.tight_layout()

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

if output_file is not None:
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
plt.close('all')
