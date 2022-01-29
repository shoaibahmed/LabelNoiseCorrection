#!/bin/bash

python3 plot_data.py \
    ./noise_models_PreResNet18_CIFAR10-CE-Baseline/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Tol-2-Thresh-20/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Stop-Train-Baseline-Tol-2-Thresh-20/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep-One-Std-Below/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-Tol-2-Thresh-20/ \
    F \
    output_plot_30_01_22.png

python3 plot_data.py \
    ./noise_models_PreResNet18_CIFAR10-CE-Baseline/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Tol-2-Thresh-20/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Stop-Train-Baseline-Tol-2-Thresh-20/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep-One-Std-Below/ \
    ./noise_models_PreResNet18_CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-Tol-2-Thresh-20/ \
    T \
    output_plot_baseline_30_01_22.png
