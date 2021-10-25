#!/bin/bash

for noise_level in 0 20 50 70 80 90; do
    ### CE loss
    python3 train.py --Mixup 'None' --experiment-name 'CE-Noise-'${noise_level} --BootBeta None \
        --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/

    ### CE loss with flooding
    /opt/conda/bin/python train.py --Mixup 'None' --experiment-name 'CE-Flooding-Noise-'${noise_level} --flood-test --BootBeta None \
        --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/
done
