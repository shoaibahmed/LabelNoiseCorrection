#!/bin/bash
# Running things

### CE loss
python3 train.py --Mixup 'None' --experiment-name 'CE' --BootBeta None \
	--epochs 300 --M 100 250 --noise-level 90 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/

### CE loss with flooding
/opt/conda/bin/python train.py --Mixup 'None' --experiment-name 'CE-Flooding' --flood-test --BootBeta None \
	--epochs 300 --M 100 250 --noise-level 90 --dataset CIFAR10 --root-dir /netscratch/siddiqui/Repositories/data/cifar10/

exit

# ####### Joint Mixup and Bootstrapping #######

python3 train.py --h

### Static mixup with hard bootstrapping (M-DYR-H)
python3 train.py --Mixup 'Static' --BootBeta 'Hard' --experiment-name 'M-DYR-H' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

### Static mixup with soft bootstrapping (M-DYR-S)
python3 train.py --Mixup 'Static' --BootBeta 'Soft' --experiment-name 'M-DYR-S' \
	--epochs 300 --M 100 250 --noise-level 80 --reg-term 1.0 --dataset CIFAR10 --root-dir /PATH/TO/CIFAR/

### Dynamic mixup with soft to hard bootstrapping (MD-DYR-SH)
python3 train.py --Mixup 'Dynamic' --experiment-name 'MD-DYR-SH' \
	--epochs 300 --M 100 250 --noise-level 90 --reg-term 1.0 --dataset CIFAR10--root-dir /PATH/TO/CIFAR/



