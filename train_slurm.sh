#!/bin/bash

for noise_level in 0 20 50 70 80 90; do
    ### CE loss
    srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=2 --mem=24G \
        --kill-on-bad-exit --job-name ce-noise-${noise_level} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CE" --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/ce_noise_${noise_level}.log 2>&1 &

    ### CE loss with flooding
    # /opt/conda/bin/python train.py --Mixup 'None' --experiment-name "CE-Flooding" --flood-test --BootBeta None \
    #     --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/
    srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=2 --mem=24G \
        --kill-on-bad-exit --job-name ce-flooding-noise-${noise_level} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CE-Flooding" --flood-test --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/ce_flooding_noise_${noise_level}.log 2>&1 &
done
