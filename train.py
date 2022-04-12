# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import PreResNet
import math
import torchvision.models as models
import random
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import sys
sys.path.append('../')
from utils import *


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: list) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, dataset_probe_identity):
        self.dataset = dataset
        self.dataset_probe_identity = dataset_probe_identity
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx


# def test_tensor(model, data, target, msg=None):
#     assert torch.is_tensor(data) and torch.is_tensor(target)
#     criterion = nn.CrossEntropyLoss()
    
#     model.eval()
#     with torch.no_grad():
#         output = model(data)
#         loss_vals = criterion(output, target)
#         test_loss = float(loss_vals.mean())
#         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#         correct = pred.eq(target.view_as(pred)).sum().item()
#         total = len(data)
    
#     test_acc = 100. * correct / total
#     output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total, 
#                        loss_vals=loss_vals.detach().cpu().numpy())
    
#     header = "Test set" if msg is None else msg
#     print(f"{header} | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")
    
#     return output_dict


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='dataset to train on, default: CIFAR10')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA support')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed, set it to go to determinist mode. We used 1 for the paper, default: None')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')
    parser.add_argument('--noise-level', type=float, default=80.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    parser.add_argument('--experiment-name', type=str, default='runs',
                        help='name of the experiment for the output files storage, default: runs')
    parser.add_argument('--alpha', type=float, default=32, help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[100, 250],
                        help="Milestones for the LR sheduler, default 100 250")
    parser.add_argument('--Mixup', type=str, default='None', choices=['None', 'Static', 'Dynamic', 'Flooding'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'HardProbes', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")
    
    parser.add_argument('--flood-test', default=False, action='store_true', 
                        help="Use flooding-based training")
    parser.add_argument('--threshold-test', default=False, action='store_true', 
                        help="Use flooding-based training only after the threshold is met")
    parser.add_argument('--use-one-std-below-noisy-loss', default=False, action='store_true', 
                        help="Use one standard deviation below the mean loss on the noisy probes as the flooding threshold")
    parser.add_argument('--stop-training', default=False, action='store_true', 
                        help="Stop training on the examples above the loss value instead of flooding their loss")
    parser.add_argument('--dynamic-flood-thresh', default=False, action='store_true', 
                        help="Use dynamic flooding threshold during training")
    parser.add_argument('--ssl-training', default=False, action='store_true', 
                        help="Use SSL training in conjuction with the CE loss -- specifically useful for noisy samples")
    
    parser.add_argument('--std-lambda', type=float, default=0., 
                        help="Parameter of the probes std dev to be used for adjusting the threshold value, default: 0.")
    parser.add_argument('--use-mislabeled-examples', default=False, action='store_true', 
                        help="Use mislabeled examples instead of the noisy probes to identify the correct point to stop model training")
    parser.add_argument('--bootstrap-epochs', type=int, default=None, 
                        help="Number of epochs for the model to be trained conventionally (without label correction) -- defaults to 105 epochs.")
    
    args = parser.parse_args()
    
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.seed:
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed)  # CPU seed
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

        random.seed(args.seed)  # python seed for image transformation

    # CIFAR meta
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR10(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR100(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError
    ssl_training = args.ssl_training
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    model = PreResNet.ResNet18(num_classes=num_classes, ssl_training=ssl_training).to(device)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    labels = get_data_cifar_2(train_loader_track)  # it should be "clonning"
    noisy_labels = add_noise_cifar_w(train_loader, args.noise_level)  # it changes the labels in the train loader directly
    noisy_labels_track = add_noise_cifar_w(train_loader_track, args.noise_level)
    
    noisy_labels = torch.Tensor(noisy_labels)
    misclassified_instances = labels != noisy_labels
    print(f"Percentage of changed instances: {torch.sum(misclassified_instances)/float(len(misclassified_instances))*100.:2f}% ({torch.sum(misclassified_instances)}/{len(misclassified_instances)}) / Noise: {args.noise_level}")
    
    assert not args.dynamic_flood_thresh or args.flood_test
    if args.flood_test:
        assert args.reg_term == 0.
        assert args.BootBeta == "None"
        assert args.Mixup in ["None", "Flooding"]
        print("Using flooding test...")

    # path where experiments are saved
    exp_path = os.path.join('./', 'noise_models_PreResNet18_{0}'.format(args.experiment_name), str(args.noise_level))

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)
    else:
        # Check if the experiment completed or not
        files = os.listdir(exp_path)
        last_epoch_exists = any([x.startswith("last_epoch") for x in files])
        print("Contents in existing output directory:", files)
        if last_epoch_exists:
            print("Output directory already exists and the experiment seems to be completed. Skipping experiment...")
            exit()
        else:
            # Remove the old directory and recreate it
            os.rmdir(exp_path)
            os.makedirs(exp_path)
            print("Recreated the output directory after deleting old results...")
    bmm_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0

    bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
    guidedMixup_ep = 106

    if args.Mixup == 'Dynamic':
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        if args.bootstrap_epochs is not None:
            bootstrap_ep_mixup = int(args.bootstrap_epochs) + 1
        else:
            bootstrap_ep_mixup = milestones[0] + 5 + 1
    print("Using bootstrap epochs to be:", bootstrap_ep_mixup)

    countTemp = 1

    temp_length = 200 - bootstrap_ep_mixup
    
    probes = None
    
    threshold = 20
    tolerance = 2
    current_iter = 0
    current_loss_thresh = None
    threshold_test = args.threshold_test
    mixup_only_when_flooding = False
    test_detection_performance = False
    use_one_std_below_noisy_loss = args.use_one_std_below_noisy_loss
    stop_training = args.stop_training
    use_probes = "Probes" in args.BootBeta
    
    if args.flood_test or test_detection_performance or use_probes:
        probes = {}
        tensor_shape = (3, 32, 32)  # For both CIFAR-10/100
        num_example_probes = 250  # 0.5% of the dataset
        normalizer = transforms.Normalize(mean, std)
        
        # probes["noisy"] = torch.clamp(torch.randn(num_example_probes, *tensor_shape), 0., 1.)
        if args.use_mislabeled_examples:
            print("Using examples from the dataset with random labels as probe...")
            selected_indices = np.random.choice(np.arange(len(trainset)), size=num_example_probes, replace=False)
            assert len(np.unique(selected_indices)) == len(selected_indices)
            
            transforms_clean = transforms.ToTensor()
            images = [train_loader.sampler.data_source.data[i] for i in selected_indices]
            probes["noisy"] = torch.stack([transforms_clean(x) for x in images], dim=0)
            print("Selected image shape:", probes["noisy"].shape)
            
            # Remove these examples from the dataset
            print(f"Dataset before deletion: {len(trainset)} / Dataloader size: {len(train_loader)}")
            num_total_examples = len(trainset)
            trainset.data = [trainset.data[i] for i in range(num_total_examples) if i not in selected_indices]
            trainset.targets = [trainset.targets[i] for i in range(num_total_examples) if i not in selected_indices]
            misclassified_instances = [misclassified_instances[i] for i in range(num_total_examples) if i not in selected_indices]
            
            # Reinitialize the dataloader to generate the right indices for sampler
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
            print(f"Dataset after deletion: {len(trainset)} / Dataloader size: {len(train_loader)}")
        
        else:
            print("Creating random examples with random labels as probe...")
            probes["noisy"] = torch.empty(num_example_probes, *tensor_shape).uniform_(0., 1.)
        
        assert probes["noisy"].shape == (num_example_probes, *tensor_shape)
        probes["noisy"] = normalizer(probes["noisy"]).to(device)
        probes["noisy_labels"] = torch.randint(0, num_classes, (num_example_probes,)).to(device)
        
        probe_images = torch.cat([probes["noisy"]], dim=0)
        probe_labels = torch.cat([probes["noisy_labels"]], dim=0)
        probe_dataset_standard = CustomTensorDataset(probe_images.to("cpu"), [int(x) for x in probe_labels.to("cpu").numpy().tolist()])
        comb_trainset = torch.utils.data.ConcatDataset([trainset, probe_dataset_standard])
        
        probe_identity = ["noisy_probe" for _ in range(len(probe_images))]
        dataset_probe_identity = ["train_noisy" if misclassified_instances[i] else "train_clean" for i in range(len(trainset))] + probe_identity
        assert len(dataset_probe_identity) == len(comb_trainset)
        print("Probe dataset:", len(comb_trainset), comb_trainset[0][0].shape)
        
        idx_dataset = IdxDataset(comb_trainset, dataset_probe_identity)
        idx_train_loader = torch.utils.data.DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        train_loader_w_probes = torch.utils.data.DataLoader(comb_trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
        
        total_instances = len(idx_dataset)
        noisy_probe_instances = np.sum([1 if dataset_probe_identity[i] == "noisy_probe" else 0 for i in range(len(idx_dataset))])
        noisy_train_instances = np.sum([1 if dataset_probe_identity[i] == "train_noisy" else 0 for i in range(len(idx_dataset))])
        clean_train_instances = np.sum([1 if dataset_probe_identity[i] == "train_clean" else 0 for i in range(len(idx_dataset))])
        print(f"Total instances: {total_instances} / Noisy probe instances: {noisy_probe_instances} / Noisy train instances: {noisy_train_instances} / Clean train instances: {clean_train_instances}")

    probe_detection_list = []
    bmm_detection_list = []

    for epoch in range(1, args.epochs + 1):
        # train
        scheduler.step()
        
        ### Standard CE training (without mixup) ###
        if args.Mixup in ["None", "Flooding"]:
            if args.flood_test:
                if args.Mixup == "Flooding":
                    assert not args.stop_training
                    alpha = 32
                    mixup_alpha = (1. / alpha) if current_loss_thresh is None and mixup_only_when_flooding else alpha
                    print(f"\t##### Doing standard training with mix-up (alpha={mixup_alpha}) loss and {'dynamic ' if args.dynamic_flood_thresh else ''}flooding #####")
                    (loss_per_epoch, acc_train_per_epoch_i), (example_idx, predictions, targets) = train_mixUp_probes(args, model, device, idx_train_loader, optimizer, epoch, mixup_alpha, current_loss_thresh)
                else:
                    if stop_training:
                        print(f"\t##### Doing standard training with cross-entropy loss and {'dynamic ' if args.dynamic_flood_thresh else ''}threshold for stopping training #####")
                    else:
                        print(f"\t##### Doing standard training with cross-entropy loss and {'dynamic ' if args.dynamic_flood_thresh else ''}flooding #####")
                    (loss_per_epoch, acc_train_per_epoch_i), (example_idx, predictions, targets) = train_CrossEntropy_probes(args, model, device, idx_train_loader, optimizer, epoch, current_loss_thresh,
                                                                                                                             use_ex_weights=stop_training, stop_learning=stop_training, use_ssl=ssl_training)
                noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
                
                # Compute loss thresh
                start_flooding = epoch > 2  # Start flooding from 3rd epoch onwards
                if args.dynamic_flood_thresh and start_flooding:
                    if threshold_test:
                        if current_loss_thresh is None:
                            if float(noisy_stats["acc"]) > threshold:
                                print(f"Noisy data accuracy ({noisy_stats['acc']}%) exceeded threshold ({threshold}%). Increasing tolerance counter...")
                                current_iter += 1
                                if current_iter >= tolerance:
                                    current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
                                    print(f"Enabling dynamic flooding barrier for the first time. Flooding loss threshold selected to be: {current_loss_thresh}")
                            else:
                                current_iter = 0
                        else:
                            # current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
                            # print(f"Using dynamic flooding barrier. Flooding loss threshold selected to be: {current_loss_thresh}")
                            if use_one_std_below_noisy_loss:
                                loss_mean, loss_std = np.mean(noisy_stats["loss_vals"]), np.std(noisy_stats["loss_vals"])
                                current_loss_thresh = np.maximum(loss_mean - loss_std, 0.)
                                print(f"Using threshold-based dynamic flooding barrier. Loss vals: (mean: {loss_mean} / std: {loss_std}). Flooding loss threshold selected to be: {current_loss_thresh}")
                            else:
                                current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
                                print(f"Using threshold-based dynamic flooding barrier. Flooding loss threshold selected to be: {current_loss_thresh}")
                    else:
                        if use_one_std_below_noisy_loss:
                            loss_mean, loss_std = np.mean(noisy_stats["loss_vals"]), np.std(noisy_stats["loss_vals"])
                            current_loss_thresh = np.maximum(loss_mean - loss_std, 0.)
                            print(f"Using consistent dynamic flooding barrier. Loss vals: (mean: {loss_mean} / std: {loss_std}). Flooding loss threshold selected to be: {current_loss_thresh}")
                        else:
                            current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
                            print(f"Using consistent dynamic flooding barrier. Flooding loss threshold selected to be: {current_loss_thresh}")
                else:
                    if current_loss_thresh is None:
                        if float(noisy_stats["acc"]) > threshold:
                            print(f"Noisy data accuracy ({noisy_stats['acc']}%) exceeded threshold ({threshold}%). Increasing tolerance counter...")
                            current_iter += 1
                            if current_iter >= tolerance:
                                # current_loss_thresh = noisy_stats["loss"] * 0.8  # 80% of the average loss on the noisy probes
                                current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
                                print(f"Enabling flooding barrier. Flooding loss threshold selected to be: {current_loss_thresh}")
                        else:
                            current_iter = 0
            else:
                print('\t##### Doing standard training with cross-entropy loss #####')
                loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, use_ssl=ssl_training)

        ### Mixup ###
        if args.Mixup == "Static":
            # TODO: Include it here
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            else:
                if args.BootBeta == "Hard":
                    print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,\
                                                                                     alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)
                if args.BootBeta == "HardProbes":
                    print("\t##### Doing HARD BETA bootstrapping with Probes and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta_probes(args, model, device, train_loader_w_probes, optimizer, epoch,\
                                                                                     alpha, args.reg_term, num_classes, probes, args.std_lambda)
                elif args.BootBeta == "Soft":
                    print("\t##### Doing SOFT BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, \
                                                                                     alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)

        ## Dynamic Mixup ##
        if args.Mixup == "Dynamic":
            alpha = args.alpha
            if epoch < guidedMixup_ep:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(guidedMixup_ep - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            elif epoch < bootstrap_ep_mixup:
                print('\t##### Doing Dynamic mixup from epoch {0} #####'.format(guidedMixup_ep))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,\
                                                                        bmm_model_maxLoss, bmm_model_minLoss)
            else:
                print("\t##### Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i, countTemp, k = train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, \
                                                                                                     epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, \
                                                                                                     countTemp, k, temp_length, args.reg_term, num_classes)
        if args.Mixup != "None" or test_detection_performance:
            ### Training tracking loss
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, model, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

        # test
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont>0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont+=1

        if epoch == args.epochs:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestValLoss_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))
        
        if test_detection_performance:
            model.eval()
            
            noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
            current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            for detector in ["probe", "bmm"]:
                tp, fp, tn, fn = 0, 0, 0, 0
                for batch_idx, ((data, target), ex_idx) in enumerate(idx_train_loader):
                    data, target = data.to(device), target.to(device)
                    if detector == "probe":
                        with torch.no_grad():
                            output = model(data)
                            loss = criterion(output, target)
                            is_misclassified_instance_probe = loss >= current_loss_thresh
                    else:
                        noisy_prob = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
                        noisy_prob = noisy_prob.to(device)
                        is_misclassified_instance_probe = noisy_prob >= 0.5  # Using 0.5 as the threshold
                    
                    for i in range(len(is_misclassified_instance_probe)):
                        ex_dataset_idx = ex_idx[i]
                        is_predicted_noisy = is_misclassified_instance_probe[i]
                        if dataset_probe_identity[ex_dataset_idx] == "train_noisy":
                            if is_predicted_noisy:
                                tp += 1
                            else:
                                fn += 1
                        elif dataset_probe_identity[ex_dataset_idx] == "train_clean":
                            if is_predicted_noisy:
                                fp += 1
                            else:
                                tn += 1
                        else:
                            assert dataset_probe_identity[ex_dataset_idx] == "noisy_probe"
                
                if tp + fp == 0:
                    detection_precision = 100.
                else:
                    detection_precision = 100. * float(tp) / (tp + fp)
                if tp + fn == 0:
                    detection_recall = 100.
                else:
                    detection_recall = 100. * float(tp) / (tp + fn)
                if detection_precision + detection_recall == 0.:
                    detection_fscore = 0.
                else:
                    detection_fscore = (2 * detection_precision * detection_recall) / (detection_precision + detection_recall)
                print(f"Detector: {detector.upper()} / TP: {tp} / FP: {fp} / TN: {tn} / FN: {fn}")
                print(f"Precision: {detection_precision:.2f}% / Recall: {detection_recall:.2f}% / F-Measure: {detection_fscore:.2f}")
                if detector == "probe":
                    probe_detection_list.append((detection_precision, detection_recall, detection_fscore))
                else:
                    assert detector == "bmm"
                    bmm_detection_list.append((detection_precision, detection_recall, detection_fscore))
            
            model.train()
    
    if test_detection_performance:
        assert len(probe_detection_list) == len(bmm_detection_list)
        with open(os.path.join(exp_path, "stats.pkl"), "wb") as f:
            stats_dict = {"probes": probe_detection_list, "bmm": bmm_detection_list}
            pickle.dump(stats_dict, f)
        
        # line_styles = ["solid", "dashed", "dashdor", "dotted"]
        # marker_list = ["o", "*", "X", "P", "D", "v", "^", "h", "1", "2", "3", "4"]
        # cm = plt.get_cmap("rainbow")
        # num_colors = 8
        # marker_colors = [cm(1.*i/num_colors) for i in range(num_colors)]
        
        plt.figure(figsize=(15, 8))
        # for i in range(len(probe_detection_list)):
        #     prec_probe, recall_probe, fscore_probe = probe_detection_list[i]
        #     prec_bmm, recall_bmm, fscore_bmm = bmm_detection_list[i]
        x_axis = list(range(1, len(probe_detection_list)+1))
        plt.plot(x_axis, [x[0] for x in probe_detection_list], label="Precision (Probes)")
        plt.plot(x_axis, [x[1] for x in probe_detection_list], label="Recall (Probes)")
        plt.plot(x_axis, [x[2] for x in probe_detection_list], label="F-Measure (Probes)")
        
        plt.plot(x_axis, [x[0] for x in bmm_detection_list], label="Precision (BMM)")
        plt.plot(x_axis, [x[1] for x in bmm_detection_list], label="Recall (BMM)")
        plt.plot(x_axis, [x[2] for x in bmm_detection_list], label="F-Measure (BMM)")
        
        plt.xlabel("Epochs")
        plt.ylabel("Percentage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_path, "detection_results.png"), dpi=300)
        
        with open(os.path.join(exp_path, "stats.csv"), "w") as f:
            f.write(f"Epoch,Probe Precision,Probe Recall,Probe F-Score,BMM Precision,BMM Recall,BMM F-Score\n")
            for epoch in range(len(probe_detection_list)):
                probe_s = probe_detection_list[epoch]
                bmm_s = bmm_detection_list[epoch]
                f.write(f"{epoch},{probe_s[0]},{probe_s[1]},{probe_s[2]},{bmm_s[0]},{bmm_s[1]},{bmm_s[2]}\n")
            
            x_axis = [i for i in range(1, len(probe_detection_list)+1)]
            auc_probe_p = auc(x_axis, [x[0] for x in probe_detection_list])
            auc_probe_r = auc(x_axis, [x[1] for x in probe_detection_list])
            auc_probe_f = auc(x_axis, [x[2] for x in probe_detection_list])
            
            auc_bmm_p = auc(x_axis, [x[0] for x in bmm_detection_list])
            auc_bmm_r = auc(x_axis, [x[1] for x in bmm_detection_list])
            auc_bmm_f = auc(x_axis, [x[2] for x in bmm_detection_list])
            
            f.write(f"-1,{auc_probe_p},{auc_probe_r},{auc_probe_f},{auc_bmm_p},{auc_bmm_r},{auc_bmm_f}\n")


if __name__ == '__main__':
    main()
