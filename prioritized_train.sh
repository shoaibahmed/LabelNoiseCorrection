#!/bin/bash

for seed in 1; do
    # Uniform batch selection
    job=c1m_ce_seed_320_bs32_adamw_${seed}
    srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-320-32BS-AdamW-CE" --BootBeta None \
            --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 1 \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Uniform batch selection, but with ImageNet pretrained model
    job=c1m_ce_seed_320_bs32_im_pretrained_adamw_${seed}
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-320-32BS-Im-Pretrained-AdamW-CE" --BootBeta None \
            --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Save loss trajectories with full training (only required once before online batch selection based on loss trajectories)
    # job=c1m_ce_seed_bs128_loss_traj_full_adamw_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=100G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-128BS-AdamW-CE-Full-Traj" --BootBeta "Probes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 128 --selection-batch-size 32 --test-batch-size 128 \
    #         --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Use loss trajectories for online batch selection
    job=c1m_m_dyr_h_probe_traj_prob_320_bs32_adamw_seed_${seed}
    srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-320-32BS-AdamW-Seed-${seed}" --BootBeta "HardProbes" \
            --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 \
            --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj/clothing1m_loss_trajectories_ep_25.pkl \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Use loss trajectories for online batch selection, but with ImageNet pretrained model
    job=c1m_m_dyr_h_probe_traj_prob_320_bs32_im_pretrained_adamw_seed_${seed}
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-320-32BS-Im-Pretrained-AdamW-Seed-${seed}" --BootBeta "HardProbes" \
            --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
            --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj/clothing1m_loss_trajectories_ep_25.pkl \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &
done
