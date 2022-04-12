#!/bin/bash

# python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
# 	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

# for noise_level in 0 20 50 70 80 90; do
#     # ### MixUp with hard bootsraping and probes for noisy example identification 
#     # job=cifar10_noise_${noise_level}_m_dyr_h_probes
#     # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#     #     --kill-on-bad-exit --job-name ${job} --nice=0 \
#     #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes" --BootBeta "HardProbes" \
#     #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#     #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &

#     ### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
#     for lambda in -1.0 -0.5 0.0 0.5 1.0; do
#         job=cifar10_noise_${noise_level}_m_dyr_h_probes_mislabeled_std_$lambda
#         srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#             --kill-on-bad-exit --job-name ${job} --nice=0 \
#             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#             /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes-Mislabeled-Std_"$lambda --BootBeta "HardProbes" \
#                 --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#                 --use-mislabeled-examples --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
    
#         job=cifar10_noise_${noise_level}_m_dyr_h_probes_noisy_std_$lambda
#         srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#             --kill-on-bad-exit --job-name ${job} --nice=0 \
#             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#             /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes-Noisy-Std_"$lambda --BootBeta "HardProbes" \
#                 --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#                 --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
#     done

#     # ### M-DYR-H config (without regularization)
#     # job=cifar10_noise_${noise_level}_m_dyr_h_paper
#     # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#     #     --kill-on-bad-exit --job-name ${job} --nice=0 \
#     #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H" --BootBeta "Hard" \
#     #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#     #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &

#     # ### Original M-DYR-H config from the paper with regularization
#     # job=cifar10_noise_${noise_level}_m_dyr_h_orig
#     # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#     #     --kill-on-bad-exit --job-name ${job} --nice=0 \
#     #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#     #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#     #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Orig" --BootBeta "Hard" \
#     #         --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
#     #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
# done

for noise_level in 0 20 50 70 80 90; do
    ### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
    bootstrapepochs=10
    for lambda in -0.5; do
        job=cifar10_noise_${noise_level}_m_dyr_h_probes_mislabeled_std_${lambda}_bootstrap_${bootstrapepochs}
        srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes-Mislabeled-Std_"$lambda --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --use-mislabeled-examples --bootstrap-epochs ${bootstrapepochs} --std-lambda ${lambda} > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
    
        job=cifar10_noise_${noise_level}_m_dyr_h_probes_noisy_std_${lambda}_bootstrap_${bootstrapepochs}
        srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Probes-Noisy-Std_"$lambda --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ \
                --bootstrap-epochs ${bootstrapepochs} --std-lambda ${lambda} > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs/${job}.log 2>&1 &
    done

done
