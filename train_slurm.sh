#!/bin/bash

for noise_level in 0 20 50 70 80 90; do
#     ### CE loss
#     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=2 --mem=24G \
#         --kill-on-bad-exit --job-name cifar10-ce-noise-${noise_level} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE" --BootBeta None \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_noise_${noise_level}.log 2>&1 &

    # ### CE loss with flooding
    # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    #     --kill-on-bad-exit --job-name cifar10-ce-flooding-noise-${noise_level}-thresh20 --nice=0 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Flooding-Thresh-20-Avg-Loss" --flood-test --BootBeta None \
    #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_flooding_thresh_20_noise_${noise_level}_avg_loss.log 2>&1 &

    ### CE loss with flooding
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name cifar100-ce-flooding-noise-${noise_level}-thresh20 --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR100-CE-Flooding-Thresh-20-Avg-Loss" --flood-test --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar100_ce_flooding_thresh_20_noise_${noise_level}_avg_loss.log 2>&1 &

    ### CE loss with dynamic flooding
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name cifar10-ce-flooding-noise-${noise_level}-thresh20 --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Dynamic-Flooding-Thresh-20-Avg-Loss" --flood-test --BootBeta None \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --dynamic-flood-thresh \
            > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/ce_dynamic_flooding_thresh_20_noise_${noise_level}_avg_loss.log 2>&1 &
done

# for noise_level in 0 20 50 70 80 90; do
# #     ### CE loss
# #     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=2 --mem=24G \
# #         --kill-on-bad-exit --job-name cifar100-ce-noise-${noise_level} --nice=0 \
# #         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
# #         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
# #         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR100-CE" --BootBeta None \
# #             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
# #             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar100_ce_noise_${noise_level}.log 2>&1 &

#     ### CE loss with flooding
#     srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name cifar100-ce-flooding-noise-${noise_level} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR100-CE-Flooding-Thresh-50-Avg-Loss" --flood-test --BootBeta None \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar100_ce_flooding_thresh_50_noise_${noise_level}_avg_loss.log 2>&1 &
# done

# for noise_level in 0 20 50 70 80 90; do
#     ### CE loss with dynamic flooding on CIFAR-10
#     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name cifar10-ce-dynamic-flooding-noise-${noise_level} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR10-CE-Dynamic-Flooding" --flood-test --BootBeta None \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --dynamic-flood-thresh \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar10_ce_dynamic_flooding_noise_${noise_level}.log 2>&1 &

#     ### CE loss with dynamic flooding on CIFAR-100
#     srun -p batch -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name cifar100-ce-dynamic-flooding-noise-${noise_level} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'None' --experiment-name "CIFAR100-CE-Dynamic-Flooding" --flood-test --BootBeta None \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --dynamic-flood-thresh \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/cifar100_ce_dynamic_flooding_noise_${noise_level}.log 2>&1 &
# done
