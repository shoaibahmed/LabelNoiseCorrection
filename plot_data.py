import os
import sys
from natsort import natsorted
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <Model directories> <Include baselines: T/F> <Output file name>")
    exit()
# model_dirs = sys.argv[1].split(";")
model_dirs = sys.argv[1:-3]
assert len(model_dirs) >= 1
model_name_list = sys.argv[-3].split(";")
include_baseline_results = sys.argv[-2].upper()
assert include_baseline_results in ["T", "F"]
include_baseline_results = include_baseline_results == "T"
output_file = sys.argv[-1]
assert output_file.endswith(".png")
print("Input directories:", model_dirs)
print("Output file:", output_file)

assert len(model_dirs) == len(model_name_list)

all_results = {}
is_cifar100 = None
for i, model_dir in enumerate(model_dirs):
    if model_dir.startswith("#"):
        print("Ignoring commented line:", model_dir)
        continue
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
    
    # dir_name = os.path.split(model_dir)[-1].replace("noise_models_PreResNet18_", "").replace("./", "").replace("/", "")
    # dir_name = model_dir.replace("noise_models_PreResNet18_", "").replace("./", "").replace("/", "")
    if is_cifar100 is None:
        is_cifar100 = 'cifar100' in model_dir.lower()
    
    dir_name = model_name_list[i]
    print(dir_name)
    all_results[dir_name] = results
print(all_results)

# Add other baseline results
results_cifar10 = {"Reed et al. (2015)": {0.0: (94.6, 94.7), 20.0: (82.9, 86.8), 50.0: (58.4, 79.8), 80.0: (26.8, 63.3), 90.0: (17.0, 42.9)},
                   "Patrini et al. (2017)": {0.0: (94.6, 94.7), 20.0: (83.1, 86.8), 50.0: (59.4, 79.8), 80.0: (26.2, 63.3), 90.0: (18.8, 42.9)},
                   "Zhang et al. (2018)": {0.0: (95.2, 95.3), 20.0: (92.3, 95.6), 50.0: (77.6, 87.1), 80.0: (46.7, 71.6), 90.0: (43.9, 52.2)},
                   "Arazo et al. (M-DYR-H) (2019)": {0.0: (93.4, 93.6), 20.0: (93.8, 94.0), 50.0: (91.9, 92.0), 80.0: (86.6, 86.8), 90.0: (9.9, 40.8)},}
                #    "Arazo et al. (MD-DYR-SH) (2019)": {0.0: (92.7, 93.6), 20.0: (93.6, 93.8), 50.0: (90.3, 90.6), 80.0: (77.8, 82.4), 90.0: (68.7, 69.1)}}

results_cifar100 = {"Reed et al. (2015)": {0.0: (75.9, 76.1), 20.0: (62.0, 62.1), 50.0: (37.9, 46.6), 80.0: (8.9, 19.9), 90.0: (3.8, 10.2)},
                   "Patrini et al. (2017)": {0.0: (75.2, 75.4), 20.0: (61.4, 61.5), 50.0: (37.3, 46.6), 80.0: (9.0, 19.9), 90.0: (3.4, 10.2)},
                   "Zhang et al. (2018)": {0.0: (74.4, 74.8), 20.0: (66.0, 67.8), 50.0: (46.6, 57.3), 80.0: (17.6, 30.8), 90.0: (8.1, 14.6)},
                   "Arazo et al. (M-DYR-H) (2019)": {0.0: (66.2, 70.3), 20.0: (68.5, 68.7), 50.0: (58.8, 61.7), 80.0: (47.6, 48.2), 90.0: (8.6, 12.5)},}
                #    "Arazo et al. (MD-DYR-SH) (2019)": {0.0: (71.3, 73.3), 20.0: (73.4, 73.9), 50.0: (65.4, 66.1), 80.0: (35.4, 41.6), 90.0: (20.5, 24.3)}}

model_types = list(all_results.keys())
noise_levels = []
for model_type in model_types:
    noise_levels += list(all_results[model_type].keys())
noise_levels = natsorted(np.unique(noise_levels))
# noise_levels = natsorted(list(all_results[model_types[0]].keys()))
print("Noise levels:", noise_levels)

# is_cifar100 = 'cifar100' in model_types[0].lower()
plot_end_acc = False

if include_baseline_results:
    if is_cifar100:
        all_results.update(results_cifar100)
    else:
        all_results.update(results_cifar10)
    model_types = list(all_results.keys())
print("All model names:", model_types)

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
cm = plt.get_cmap('hsv')
NUM_COLORS = len(model_types)
if NUM_COLORS > 10:
    marker_colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
else:
    marker_colors = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple", "tab:olive", "tab:gray", "tab:pink", "tab:brown", "tab:cyan"]

for idx in range(len(model_types)):
    for plot_best in [False] if plot_end_acc else [True]:
        model_results = all_results[model_types[idx]]
        acc_list = []
        noise_list = []
        for noise_level in noise_levels:
            if noise_level not in model_results:
                print(f"Warning: Noise level ({noise_level}) results not found for model: {model_types[idx]}")
                continue
            acc_list.append(model_results[noise_level][1 if plot_best else 0])
            noise_list.append(noise_level)
        
        line = plt.plot(noise_list, acc_list, linewidth=4., marker=marker_list[idx % len(marker_list)],
                        color=marker_colors[idx], alpha=0.6, markeredgecolor='k', markersize=9, label=f"{model_types[idx]}{' (Best)' if plot_best and plot_end_acc else ''}")
        # line = plt.plot(noise_list, acc_list, linewidth=2., marker=marker_list[idx % len(marker_list)],
        #                 color=marker_colors[idx], alpha=0.75, markeredgecolor='k', label=model_name_list[idx])
        line[0].set_color(marker_colors[idx])
        # line[0].set_linestyle(line_styles[(idx*2+(1 if plot_best else 0)) % len(line_styles)])
        line[0].set_linestyle(line_styles[(1 if plot_best else 0) % len(line_styles)])

font_size = 16
plt.xlabel('Noise level (%)', fontsize=font_size)
plt.ylabel('Accuracy (%)', fontsize=font_size)
# plt.title(f"Results on PreAct ResNet-18 trained on CIFAR-10{'0' if is_cifar100 else ''}{' (Best)' if not plot_end_acc else ''}")
plt.legend(prop={'size': font_size})
# plt.ylim(0., 100.)
# plt.xticks(list(range(1, 5)))
plt.tight_layout()

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

# output_file = "results.png"
if output_file is not None:
    plt.savefig(output_file, dpi=300)
plt.show()
plt.close('all')
