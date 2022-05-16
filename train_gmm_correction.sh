#!/bin/bash

for noise_level in 0 20 50 70 80 90; do
    job_name="cifar10_noise_${noise_level}_m_dyr_h_gmm_treatment_adaptive_weights"
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name ${job_name} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-GMM-Treatment-Adaptive-Weight" --BootBeta "HardProbes" \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-gmm-probe-identification \
            --bootstrap-epochs 105 --use-probes-for-pretraining --use-mislabeled-examples \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_gmm_identification/${job_name}.log 2>&1 &
    
    job_name="cifar100_noise_${noise_level}_m_dyr_h_gmm_treatment_adaptive_weights"
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name ${job_name} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-GMM-Treatment-Adaptive-Weight" --BootBeta "HardProbes" \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --use-gmm-probe-identification \
            --bootstrap-epochs 105 --use-probes-for-pretraining --use-mislabeled-examples \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_gmm_identification/${job_name}.log 2>&1 &
done
