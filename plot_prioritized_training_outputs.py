import os
import sys
from natsort import natsorted
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print(f"Usage: {sys.argv[0]} <Log directories> <Model name list> <Output file name>")
    exit()

log_files = sys.argv[1:-2]
assert len(log_files) >= 1, log_files
model_name_list = sys.argv[-2].split(";")
output_file = sys.argv[-1]
assert output_file.endswith(".png")
print("Log files:", log_files)
print("Output file:", output_file)

assert len(log_files) == len(model_name_list)

all_results = {}
for i, log_file in enumerate(log_files):
    assert os.path.exists(log_file), log_file
    print("Loading log file:", log_file)
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    # Filter the lines
    selected_lines = [l for l in lines if "Test set" in l]
    # print(selected_lines)
    
    # Extract information
    total_list = []
    correct_list = []
    loss_list = []
    for l in selected_lines:
        splits = l.split(", Accuracy: ")
        assert "Average loss" in splits[0]
        assert "(" in splits[1]
        first_splits = splits[0].split(" Average loss: ")
        second_splits = splits[1].split(" (")[0]
        correct_total = second_splits.split("/")
        assert len(correct_total) == 2
        assert len(first_splits) == 2
        loss = float(first_splits[1])
        correct = int(correct_total[0])
        total = int(correct_total[1])
        print(f"Line: {l} / Loss: {loss} / Correct: {correct} / Total: {total}")
        loss_list.append(loss)
        correct_list.append(correct)
        total_list.append(total)
    
    dir_name = model_name_list[i]
    print(dir_name)
    assert dir_name not in all_results
    assert all([total_list[0] == x for x in total_list])
    acc_list = [0.] + [100. * c / t for c, t in zip(correct_list, total_list)]
    epoch_list = [0] + [i+1 for i in range(len(correct_list))]
    all_results[dir_name] = {"total": total_list, "correct": correct_list, "loss": loss_list, "acc": acc_list, "epoch": epoch_list}

print(all_results)

model_types = list(all_results.keys())
epoch_list = []
for model_type in model_types:
    epoch_list.append(len(all_results[model_type]["epoch"]))
num_epochs = np.min(epoch_list)
print("Number of epochs selected:", num_epochs)

fig, ax = plt.subplots()
fig.set_size_inches(10, 6)

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
cm = plt.get_cmap('hsv')
NUM_COLORS = len(model_types)
if NUM_COLORS > 10:
    marker_colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
else:
    marker_colors = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:olive", "tab:gray", "tab:pink", "tab:brown", "tab:cyan"]

for idx in range(len(model_types)):
    model_results = all_results[model_types[idx]]
    accs = model_results["acc"][:num_epochs]
    epochs = model_results["epoch"][:num_epochs]
    print(f"Model type: {model_types[idx]} / Acc list: {accs} / Epoch list: {epochs}")
    
    line = plt.plot(epochs, accs, linewidth=4., marker=marker_list[idx % len(marker_list)],
                    color=marker_colors[idx], alpha=0.6, markeredgecolor='k', markersize=9, label=f"{model_types[idx]}")
    line[0].set_color(marker_colors[idx])
    # line[0].set_linestyle(line_styles[% len(line_styles)])

font_size = 16
plt.xlabel('Epochs (%)', fontsize=font_size)
plt.ylabel('Test Accuracy (%)', fontsize=font_size)
plt.legend(prop={'size': font_size})
plt.ylim(60., 73.)
plt.xlim(0., num_epochs)
# plt.xticks(list(range(1, 5)))
plt.tight_layout()

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

if output_file is not None:
    plt.savefig(output_file, dpi=300)
plt.show()
plt.close('all')
