#!/bin/bash

for seed in 1; do
    # CE loss
    job=c1m_ce_seed_${seed}
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'None' --experiment-name "C1M-CE" --BootBeta None \
            --epochs 100 --M 20 40 75 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 6 --batch-size 256 --test-batch-size 256 --bootstrap-epochs 1 \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Arazo et al.
    job=c1m_noise_${noise_level}_m_dyr_h_bmm_seed_${seed}
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "C1M-M-DYR-H-BMM-Seed-${seed}" --BootBeta "Hard" \
            --epochs 100 --M 20 40 75 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 6 --batch-size 256 --test-batch-size 256 --bootstrap-epochs 1 \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Loss trajectories
    job=c1m_m_dyr_h_probe_traj_prob_seed_${seed}
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=6 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Seed-${seed}" --BootBeta "HardProbes" \
            --epochs 100 --M 20 40 75 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 6 --batch-size 256 --test-batch-size 256 --bootstrap-epochs 1 \
            --use-mislabeled-examples --use-loss-trajectories --use-probes-for-pretraining --use-gmm-probe-identification --num-example-probes 250 \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &
done
