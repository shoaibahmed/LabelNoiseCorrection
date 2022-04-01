#!/bin/bash

# python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
# 	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

for noise_level in 0 20 50 70 80 90; do
    ### CE loss with dynamic identification for stopping training 
    job=cifar10_noise_${noise_level}_hm_dyr_h
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes" --BootBeta "HardProbes" \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
done
