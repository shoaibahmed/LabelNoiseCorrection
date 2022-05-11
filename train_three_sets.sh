#!/bin/bash

# python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
# 	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

# for noise_level in 20 25 30 35 40 45; do
#     ### CE loss
#     job_name="cifar10_ce_noise_${noise_level}_three_sets_random"
#     srun -p RTX6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name ${job_name} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'None' --experiment-name "CIFAR10-CE-Three-Sets-Random" --BootBeta None \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job_name}.log 2>&1 &
    
#     # DYR formulation
#     job_name="cifar10_ce_noise_${noise_level}_m_dyr_h_three_sets_random"
#     srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name ${job_name} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Three-Sets-Random" --BootBeta "Hard" \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job_name}.log 2>&1 &

#     ### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
#     for lambda in -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0; do
#         job=cifar10_noise_${noise_level}_m_dyr_h_three_sets_random_probes_pretraining_stop_0.2_noisy_std_$lambda
#         srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#             --kill-on-bad-exit --job-name ${job} --nice=0 \
#             --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#             --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#             /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Three-Sets-Random-Probes-Pretraining-Stop-0.2-Noisy-Std_"$lambda --BootBeta "HardProbes" \
#                 --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
#                 --bootstrap-probe-acc-thresh 20.0 --use-probes-for-pretraining --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job}.log 2>&1 &
#     done

#     # GMM with probes
#     job_name="cifar10_ce_noise_${noise_level}_three_sets_random_treatment"
#     srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name ${job_name} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-CE-Three-Sets-Random-Treatment" --BootBeta "HardProbes" \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets --treat-three-sets \
#             --bootstrap-epochs 105 --use-probes-for-pretraining --bootstrap-probe-acc-thresh 20.0 \
#             > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job_name}.log 2>&1 &
# done

for noise_level in 20 25 30 35 40 45; do
    # ### CE loss
    # job_name="cifar100_ce_noise_${noise_level}_three_sets_random"
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    #     --kill-on-bad-exit --job-name ${job_name} --nice=0 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'None' --experiment-name "CIFAR100-CE-Three-Sets-Random" --BootBeta None \
    #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --use-three-sets \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_three_sets/${job_name}.log 2>&1 &
    
    # # # DYR formulation
    # job_name="cifar100_ce_noise_${noise_level}_m_dyr_h_three_sets_random"
    # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    #     --kill-on-bad-exit --job-name ${job_name} --nice=0 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-Three-Sets-Random" --BootBeta "Hard" \
    #         --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --use-three-sets \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_three_sets/${job_name}.log 2>&1 &

    # # ### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
    # for lambda in -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0; do
    for lambda in -0.5 0.5; do
        job=cifar100_noise_${noise_level}_m_dyr_h_three_sets_random_probes_pretraining_stop_0.2_noisy_std_$lambda
        srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
            --kill-on-bad-exit --job-name ${job} --nice=0 \
            --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
            --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
            /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-Three-Sets-Random-Probes-Pretraining-Stop-0.2-Noisy-Std_"$lambda --BootBeta "HardProbes" \
                --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --use-three-sets \
                --bootstrap-probe-acc-thresh 20.0 --use-probes-for-pretraining --std-lambda $lambda > \
                /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_three_sets/${job}.log 2>&1 &
    done

    # # GMM with probes
    # job_name="cifar100_ce_noise_${noise_level}_three_sets_random_treatment"
    # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    #     --kill-on-bad-exit --job-name ${job_name} --nice=0 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-CE-Three-Sets-Random-Treatment" --BootBeta "HardProbes" \
    #         --epochs 300 --M 100 250 --noise-level ${noise_level} --dataset CIFAR100 --root-dir /netscratch/siddiqui/Repositories/data/cifar100/ --use-three-sets --treat-three-sets \
    #         --bootstrap-epochs 105 --use-probes-for-pretraining --bootstrap-probe-acc-thresh 20.0 \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_three_sets/${job_name}.log 2>&1 &
done

exit

### Original M-DYR-H config from the paper with regularization
job_name="cifar10_ce_noise_${noise_level}_m_dyr_h_three_sets_random"
srun -p RTX6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
    --kill-on-bad-exit --job-name ${job_name} --nice=0 \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Three-Sets-Random" --BootBeta "Hard" \
        --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
        > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job_name}.log 2>&1 &

### MixUp with hard bootsraping and probes for noisy example identification (including high regularization)
for lambda in -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0; do
    job=cifar10_noise_${noise_level}_m_dyr_h_three_sets_random_probes_pretraining_stop_0.2_noisy_std_$lambda
    srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
        --kill-on-bad-exit --job-name ${job} --nice=0 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR10-M-DYR-H-Three-Sets-Random-Probes-Pretraining-Stop-0.2-Noisy-Std_"$lambda --BootBeta "HardProbes" \
            --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
            --bootstrap-probe-acc-thresh 20.0 --use-probes-for-pretraining --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job}.log 2>&1 &
done

# for lambda in -1.0 -0.5 0.0 0.5 1.0; do
#     job=cifar10_noise_${noise_level}_m_dyr_h_probes_noisy_std_$lambda
#     srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem=24G \
#         --kill-on-bad-exit --job-name ${job} --nice=0 \
#         --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
#         --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
#         /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/train_three_subs.py --Mixup 'Static' --experiment-name "CIFAR100-M-DYR-H-Probes-Noisy-Std_"$lambda --BootBeta "HardProbes" \
#             --epochs 300 --M 100 250 --noise-level ${noise_level} --reg-term 1.0 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/ --use-three-sets \
#             --std-lambda $lambda > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_26_04_22/${job}.log 2>&1 &
# done
