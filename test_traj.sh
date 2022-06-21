#!/bin/bash

for noise_level in 0 20 50 70 80 90; do
    for seed in 1 2 3; do
        job=cifar10_noise_${noise_level}_m_dyr_h_probe_traj_prob_seed_${seed}
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probe-Traj-Prob-Seed-${seed}" --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --use-mislabeled-examples --use-loss-trajectories --use-probes-for-pretraining --seed ${seed} \
                --use-gmm-probe-identification --bootstrap-epochs 105 --num-example-probes 250 \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_traj/${job}.log 2>&1 &
    done
done

for noise_level in 0 20 50 70 80 90; do
    for seed in 1 2 3; do
        job=cifar100_noise_${noise_level}_m_dyr_h_probe_traj_prob_seed_${seed}
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-Probe-Traj-Prob-Seed-${seed}" --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
                --use-mislabeled-examples --use-loss-trajectories --use-probes-for-pretraining --seed ${seed} \
                --use-gmm-probe-identification --bootstrap-epochs 105 --num-example-probes 250 \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_traj/${job}.log 2>&1 &
    done
done



################################## Three-set experiments ##################################

for noise_level in 25; do
    for seed in 1 2 3; do
        job=cifar10_noise_${noise_level}_m_dyr_h_three_sets_probe_traj_prob_seed_${seed}
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Three-Sets-Probe-Traj-Prob-Seed-${seed}" --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --use-mislabeled-examples --use-loss-trajectories --use-probes-for-pretraining --seed ${seed} \
                --use-gmm-probe-identification --bootstrap-epochs 105 --num-example-probes 250 --use-three-sets --treat-three-sets \
                > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_traj/${job}.log 2>&1 &
    done
done
