#!/bin/bash

# Use loss trajectories with three sets for online batch selection (with only 250 probe examples), but with ImageNet pretrained model and default AdamW WD
# seed=1
# python prioritized_train.py --Mixup 'None' --experiment-name "C1M-Probe-Traj-Not-Inc-Prob-Three-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
#         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /mnt/sas/Datasets/clothing/ --seed ${seed} \
#         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
#         --use-three-set-prioritized-training --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
#         | tee ./logs_c1m/c1m_not_inc_three_sets.log

# python prioritized_train.py --Mixup 'None' --experiment-name "C1M-Probe-Traj-Inc-Prob-Three-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
#         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /mnt/sas/Datasets/clothing/ --seed ${seed} \
#         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model --use-probes-for-pretraining \
#         --use-three-set-prioritized-training --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
#         | tee ./logs_c1m/c1m_inc_three_sets.log

for seed in 1 2 3; do
    # (First) Uniform batch selection, but with ImageNet pretrained model and default AdamW WD
    # job=c1m_ce_seed_320_bs32_im_pretrained_adamw_0.01wd_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-320-32BS-Im-Pretrained-AdamW-0.01WD-CE-Seed-${seed}" --BootBeta None \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Save loss trajectories with full training and including validation probes in training (only required once before online batch selection based on loss trajectories)
    # job=c1m_ce_bs128_loss_traj_full_val_probe_pretrain_adamw_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=100G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain-Seed-${seed}" --BootBeta "Probes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 128 --selection-batch-size 32 --test-batch-size 128 --use-probes-for-pretraining \
    #         --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # job=c1m_ce_bs128_loss_traj_full_val_probe_pretrain_adamw_0.01wd_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=100G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-128BS-AdamW-0.01WD-CE-Full-Traj-Val-Probe-Pretrain-Seed-${seed}" --BootBeta "Probes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 128 --selection-batch-size 32 --test-batch-size 128 --use-probes-for-pretraining \
    #         --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories for online batch selection, but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_traj_prob_probe_pretrain_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Probe-Pretrain-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "HardProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_25.pkl \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use only loss value for online batch selection, but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_traj_prob_probe_pretrain_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Probe-Pretrain-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "HardProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_25.pkl \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories + correct class confidence for online batch selection, but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_traj_prob_conf_score_probe_pretrain_5ep_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Conf-Score-Probe-Pretrain-5ep-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories + correct class confidence for online batch selection (with only 250 probe examples), but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_traj_prob_conf_score_probe_pretrain_5ep_limit250_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Conf-Score-Probe-Pretrain-5ep-Limit250-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories + correct class confidence for online batch selection (with only 250 probe examples), but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_traj_prob_conf_score_probe_pretrain_5ep_limit250_320_bs32_batch_cls_balanced_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Conf-Score-Probe-Pretrain-5ep-Limit250-320-32BS-Batch-Cls-Balanced-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # Use online loss value for online batch selection (with only 250 probe examples), but with ImageNet pretrained model and default AdamW WD
    # job=c1m_m_dyr_h_probe_online_typical_320_bs32_batch_cls_balanced_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p V100-32GB -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Typical-320-32BS-Batch-Cls-Balanced-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # job=c1m_m_dyr_h_probe_online_typical_train_m_320_bs32_batch_cls_balanced_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p V100-32GB -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Typical-Train-M-320-32BS-Batch-Cls-Balanced-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # job=c1m_m_dyr_h_probe_online_dist_0.2percentile_typical_train_320_bs32_batch_not_balanced_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p A100-SDS -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Dist-Min-Typical-Train-320-32BS-Batch-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-Probe-Traj-3ep-Recompute-Train-320BS-Im-Pretrained-AdamW-0.01WD-Seed-1/c1m_loss_trajectories_ep_3_recompute_train.pkl \
    #         --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # SAS - Main (14 / 08 / 22)
    # job=c1m_m_dyr_h_probe_online_score_sample_320_bs32_batch_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Score-Sample-320-32BS-Batch-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-Probe-Traj-3ep-Recompute-Train-320BS-Im-Pretrained-AdamW-0.01WD-Seed-1/c1m_loss_trajectories_ep_3_recompute_train.pkl \
    #         --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # SAS - Main (14 / 08 / 22)
    # job=c1m_m_dyr_h_probe_online_score_sample_balanced_320_bs32_batch_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Score-Balanced-Sample-320-32BS-Batch-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-Probe-Traj-3ep-Recompute-Train-320BS-Im-Pretrained-AdamW-0.01WD-Seed-1/c1m_loss_trajectories_ep_3_recompute_train.pkl \
    #         --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # job=c1m_m_dyr_h_probe_online_dist_typical_train_320_bs32_batch_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p A100-SDS -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Online-Dist-Typical-Train-320-32BS-Batch-Cls-Balanced-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 --use-probes-for-pretraining --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification \
    #         --num-example-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # Use correct class confidence for online batch selection, but with ImageNet pretrained model and default AdamW WD
    # SAS - Main (14 / 08 / 22)
    job=c1m_correct_score_320_bs32_batch_im_pretrained_adamw_0.01wd_seed_${seed}
    srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
        --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
        --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-Correct-Score-320-32BS-Batch-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
            --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
            --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
            --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
            --subsample-val-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m_new/${job}.log 2>&1 &

    # SAS - Main (14 / 08 / 22)
    # job=c1m_uniform_balanced_cls_batch_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-Uniform-Balanced-Cls-Batch-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # job=c1m_loss_score_batch_320_bs32_im_pretrained_adamw_0.01wd_seed_${seed}
    # srun -p RTX3090 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-Loss-Score-Batch-320-32BS-Im-Pretrained-AdamW-0.01WD-Seed-${seed}" --BootBeta "RHOProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-2 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_5.pkl \
    #         --subsample-val-probes 250 > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &
done
exit

# old
for seed in 1; do
    # Uniform batch selection
    # job=c1m_ce_seed_320_bs32_adamw_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-320-32BS-AdamW-CE-Seed-${seed}" --BootBeta None \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 1 \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Uniform batch selection, but with ImageNet pretrained model
    # job=c1m_ce_seed_320_bs32_im_pretrained_adamw_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-320-32BS-Im-Pretrained-AdamW-CE-Seed-${seed}" --BootBeta None \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Save loss trajectories with full training (only required once before online batch selection based on loss trajectories)
    # # job=c1m_ce_seed_bs128_loss_traj_full_adamw_${seed}
    # # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=100G \
    # #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    # #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    # #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    # #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-128BS-AdamW-CE-Full-Traj" --BootBeta "Probes" \
    # #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    # #         --noise-level 0.0 --num-workers 8 --batch-size 128 --selection-batch-size 32 --test-batch-size 128 \
    # #         --use-mislabeled-examples --use-loss-trajectories --use-gmm-probe-identification --bootstrap-epochs 0 --num-example-probes 250 \
    # #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories for online batch selection
    # job=c1m_m_dyr_h_probe_traj_prob_320_bs32_adamw_seed_${seed}
    # srun -p A100 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-320-32BS-AdamW-Seed-${seed}" --BootBeta "HardProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj/clothing1m_loss_trajectories_ep_25.pkl \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

    # # Use loss trajectories for online batch selection, but with ImageNet pretrained model
    # job=c1m_m_dyr_h_probe_traj_prob_probe_pretrain_320_bs32_im_pretrained_adamw_seed_${seed}
    # srun -p RTXA6000 -K -N1 --ntasks-per-node=1 --gpus-per-task=1 --cpus-per-gpu=8 --mem=40G \
    #     --kill-on-bad-exit --job-name ${job} --nice=0 --time=3-00:00:00 \
    #     --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.06-py3.sqsh \
    #     --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
    #     /opt/conda/bin/python /netscratch/siddiqui/Repositories/LabelNoiseCorrection/prioritized_train.py --Mixup 'None' --experiment-name "C1M-M-DYR-H-Probe-Traj-Prob-Probe-Pretrain-320-32BS-Im-Pretrained-AdamW-Seed-${seed}" --BootBeta "HardProbes" \
    #         --epochs 50 --no-milestones --lr 0.001 --optimizer "adamw" --weight-decay 1e-4 --reg 0.0 --dataset Clothing1M --root-dir /netscratch/siddiqui/Datasets/clothing/ --seed ${seed} \
    #         --noise-level 0.0 --num-workers 8 --batch-size 320 --selection-batch-size 32 --test-batch-size 128 --bootstrap-epochs 0 --use-im-pretrained-model \
    #         --loss-trajectories-path ./prioritized_training_PreResNet18_C1M-128BS-AdamW-CE-Full-Traj-Val-Probe-Pretrain/clothing1m_loss_trajectories_ep_25.pkl \
    #         > /netscratch/siddiqui/Repositories/LabelNoiseCorrection/logs_c1m/${job}.log 2>&1 &

done
