#!/bin/bash

# python3 plot_data.py \
#     ./noise_models_PreResNet18_CIFAR10-CE-Baseline/ \
#     ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Tol-2-Thresh-20/ \
#     ./noise_models_PreResNet18_CIFAR10-CE-SSL-Proj-Baseline \
#     noise_models_PreResNet18_CIFAR10-CE-Stop-Train-SSL-Proj-Baseline-Tol-2-Thresh-20 \
#     F \
#     output_plot_05_02_22.png
# exit

for baselines in "F" "T"; do
    output_file=output_plot
    if [[ "$baselines" == "T" ]]; then
        output_file=${output_file}_baseline
    fi
    output_file=${output_file}.png
    echo "Writing output to file: "${output_file}

    python3 plot_data.py \
        ./noise_models_PreResNet18_CIFAR10-CE-Baseline/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Tol-2-Thresh-20/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Stop-Train-Baseline-Tol-2-Thresh-20/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep-One-Std-Below/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-Tol-2-Thresh-20/ \
        ./noise_models_PreResNet18_CIFAR10-CE-Stop-Train-Baseline-Dynamic-Thresh-3rd-Ep/ \
        ${baselines} \
        ${output_file}
done
