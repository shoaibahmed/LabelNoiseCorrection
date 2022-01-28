#!/bin/bash

for noise_level in 0 20 50 70 80 90; do

    ### CE loss with threshold-based flooding
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name cifar10-ce-flood-noise-${noise_level}-baseline-tol-2-thresh-20 --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Flood-Baseline-Tol-2-Thresh-20" --flood-test --threshold-test --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_flood_noise_${noise_level}_detection_baseline_tol_2_thresh_20.log 2>&1 &
    
    # ### CE loss with dynamic flooding after 3rd epoch onwards
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    #     --kill-on-bad-exit --job-name cifar10-ce-flood-noise-${noise_level}-baseline-dynamic-thresh-3rd-ep --nice=0 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep" --flood-test --dynamic-flood-thresh --BootBeta None \
    #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_flood_noise_${noise_level}_detection_baseline_dynamic_thresh_3rd_ep.log 2>&1 &

    ### CE loss with dynamic flooding after 3rd epoch onwards, but with the loss threshold to be one std. dev. below the mean
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name cifar10-ce-flood-noise-${noise_level}-baseline-dynamic-thresh-3rd-ep-one-std-below --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Flood-Baseline-Dynamic-Thresh-3rd-Ep-One-Std-Below" --flood-test --dynamic-flood-thresh --use-one-std-below-noisy-loss --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_flood_noise_${noise_level}_detection_baseline_dynamic_thresh_3rd_ep_one_std_below.log 2>&1 &

done
