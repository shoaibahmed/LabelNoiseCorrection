#!/bin/bash

# python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
# 	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

for noise_level in 0 20 50 70 80 90; do
    ### Original M-DYR-H config from the paper with regularization
    for seed in 1 2 3; do
        job=cifar10_noise_${noise_level}_m_dyr_h_bmm_treatment_seed_${seed}
        srun -p V100-16GB -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-BMM-Treatment-Seed-${seed}" --BootBeta "Hard" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --seed ${seed} \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_gmm_identification/${job}.log 2>&1 &
    done
done

for noise_level in 0 20 50 70 80 90; do
    for seed in 1 2 3; do
        job=cifar10_noise_${noise_level}_m_dyr_h_gmm_treatment_seed_${seed}
        srun -p V100-16GB -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-GMM-Treatment-Seed-${seed}" --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --use-gmm-probe-identification --use-mislabeled-examples --use-unmodified-train-set-for-pretraining --use-adaptive-weights --seed ${seed} \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_gmm_identification/${job}.log 2>&1 &
    done
done

for noise_level in 0 20 50 70 80 90; do
    for seed in 1 2 3; do
        job=cifar10_noise_${noise_level}_m_dyr_h_gmm_treatment_fixed_seed_${seed}
        srun -p V100-16GB -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-GMM-Treatment-Fixed-Seed-${seed}" --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --use-gmm-probe-identification --use-mislabeled-examples --use-unmodified-train-set-for-pretraining --seed ${seed} \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_gmm_identification/${job}.log 2>&1 &
    done
done
