#!/bin/bash

# python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
# 	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

for noise_level in 0 20 50 70 80 90; do
    ### Original M-DYR-H config from the paper with regularization
    job=cifar100_noise_${noise_level}_m_dyr_h
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H" --BootBeta "Hard" \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &

    ### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
    # for lambda in -1.0 -0.5 0.0 0.5 1.0; do
    for lambda in -0.5 0.5; do
        job=cifar100_noise_${noise_level}_m_dyr_h_probes_pretraining_stop_0.2_noisy_std_$lambda
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-Probes-Pretraining-Stop-0.2-Noisy-Std_"$lambda --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
                --bootstrap-probe-acc-thresh 20.0 --use-probes-for-pretraining --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_12_04_22/${job}.log 2>&1 &
    done
done
