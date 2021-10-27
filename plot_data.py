import os
import sys
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <Model directoroes>")
    exit()
model_dirs = sys.argv[1].split(";")
assert len(model_dirs) >= 1

all_results = {}
for model_dir in model_dirs:
    assert os.path.exists(model_dir)
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
        assert noise_level in [float(x) for x in range(0, 100, 10)]
        
        assert model_name_parts[9] == "bestValLoss"
        best_val_acc = float(model_name_parts[10])
        assert 0. <= best_val_acc <= 100.
        
        print(f"Noise level: {noise_level} / Val acc: {val_acc} / Best val acc: {best_val_acc}")
        results[noise_level] = (val_acc, best_val_acc)
    
    # dir_name = os.path.split(model_dir)[-1].replace("noise_models_PreResNet18_", "").replace("./", "").replace("/", "")
    dir_name = model_dir.replace("noise_models_PreResNet18_", "").replace("./", "").replace("/", "")
    print(dir_name)
    all_results[dir_name] = results
print(all_results)

model_types = list(all_results.keys())
noise_levels = natsorted(list(all_results[model_types[0]].keys()))
print("Noise levels:", noise_levels)

fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
cm = plt.get_cmap('hsv')
NUM_COLORS = len(model_types)
marker_colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

for idx in range(len(model_types)):
    for plot_best in [False, True]:
        model_results = all_results[model_types[idx]]
        acc_list = []
        for noise_level in noise_levels:
            acc_list.append(model_results[noise_level][1 if plot_best else 0])
        
        line = plt.plot(noise_levels, acc_list, linewidth=2., marker=marker_list[idx % len(marker_list)],
                        color=marker_colors[idx], alpha=0.75, markeredgecolor='k', label=f"{model_types[idx]}{' (Best)' if plot_best else ''}")
        line[0].set_color(marker_colors[idx])
        # line[0].set_linestyle(line_styles[(idx*2+(1 if plot_best else 0)) % len(line_styles)])
        line[0].set_linestyle(line_styles[(1 if plot_best else 0) % len(line_styles)])

plt.xlabel('Noise level (%)')
plt.ylabel('Accuracy (%)')
plt.title("Results on PreAct ResNet-18 trained on CIFAR-10")
plt.legend()
plt.ylim(0., 100.)
# plt.xticks(list(range(1, 5)))
plt.tight_layout()

output_file = "results.png"
if output_file is not None:
    plt.savefig(output_file, dpi=300)
plt.show()
plt.close('all')
