# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import PreResNet
import torchvision.models as models
import random
import os
import shutil
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import auc
import sys
sys.path.append('../')

from utils import *
from clothing1m import Clothing1M


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--selection-batch-size', type=int, default=None,
                        help='batch size to be used for training from the larger batch (online batch selection) -- none indicates using the full batch for training')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'Clothing1M'], help='dataset to train on, default: CIFAR10')
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
    parser.add_argument('--no-milestones', action="store_true", default=False,
                        help="Don't use LR milestones")
    parser.add_argument('--optimizer', type=str, choices=["sgd", "adam", "adamw"], default="sgd",
                        help="Optimizer to use")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument('--Mixup', type=str, default='None', choices=['None', 'Static', 'Dynamic', 'Flooding'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'HardProbes', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of worker processes for data loading')
    
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
    
    parser.add_argument('--use-gmm-probe-identification', default=False, action='store_true', 
                        help="Identify mislabeled examples using GMM coupled with probes")
    parser.add_argument('--use-adaptive-weights', default=False, action='store_true', 
                        help="Use adaptive weights for GMM")
    parser.add_argument('--use-binary-prediction', default=False, action='store_true', 
                        help="Return binary prediction from the GMM model instead of actual probabilities")
    parser.add_argument('--use-softmax', default=False, action='store_true', 
                        help="Use softmax on the probabilities returned by the model")
    parser.add_argument('--update-every-iter', default=False, action='store_true', 
                        help="Update the GMM mean and std. based on probes after every model iteration")
    parser.add_argument('--resume-from-pretraining', default=False, action='store_true', 
                        help="Resume from the pretraining phase of the model for quicker experiments")
    
    parser.add_argument('--use-three-sets', default=False, action='store_true', 
                        help="Use three sets in the dataset")
    parser.add_argument('--treat-three-sets', default=False, action='store_true', 
                        help="Treat and identify three sets in the dataset")
    parser.add_argument('--use-bmm-treatment', default=False, action='store_true', 
                        help="Use BMM to treat these three sets as compared to a GMM")
    
    parser.add_argument('--std-lambda', type=float, default=0., 
                        help="Parameter of the probes std dev to be used for adjusting the threshold value, default: 0.")
    parser.add_argument('--use-mislabeled-examples', default=False, action='store_true', 
                        help="Use mislabeled examples instead of the noisy probes to identify the correct point to stop model training")
    parser.add_argument('--use-probes-for-pretraining', default=False, action='store_true', 
                        help="Also include probes for pretraining phase.")
    parser.add_argument('--use-unmodified-train-set-for-pretraining', default=False, action='store_true', 
                        help="Also include probes for pretraining phase.")
    parser.add_argument('--bootstrap-epochs', type=int, default=None, 
                        help="Number of epochs for the model to be trained conventionally (without label correction) -- defaults to 105 epochs.")
    parser.add_argument('--bootstrap-probe-acc-thresh', type=float, default=None, 
                        help="Accuracy on the probes to be reached for the bootstraping to stop.")
    parser.add_argument('--use-loss-trajectories', default=False, action='store_true', 
                        help="Use the full loss trajectory instead of just point estimates for the loss")
    parser.add_argument('--num-example-probes', type=int, default=None, 
                        help="Number of probes to be used -- defaults to 250")
    parser.add_argument('--loss-trajectories-path', type=str, default=None, 
                        help="Path from where to load the loss trajectories")
    # parser.add_argument('--store-loss-trajectories', action="store_true", default=False, 
    #                     help="Store loss trajectories for identification")
    
    args = parser.parse_args()
    
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert not args.use_loss_trajectories or not args.use_unmodified_train_set_for_pretraining
    assert not args.use_loss_trajectories or args.use_probes_for_pretraining or args.dataset == "Clothing1M"
    assert not args.use_loss_trajectories or args.use_gmm_probe_identification or args.dataset == "Clothing1M"
    if args.loss_trajectories_path == "":
        args.loss_trajectories_path = None
    assert args.loss_trajectories_path is None or (args.dataset == "Clothing1M" and args.BootBeta == "HardProbes")
    assert args.loss_trajectories_path is None or os.path.exists(args.loss_trajectories_path), args.loss_trajectories_path
    # assert not args.store_loss_trajectories or args.dataset == "Clothing1M"
    
    if args.seed:
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed)  # CPU seed
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

        random.seed(args.seed)  # python seed for image transformation

    if args.dataset == 'Clothing1M':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform_clean = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        assert args.dataset in ['CIFAR10', 'CIFAR100']
        
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

    use_val_set = False
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
    elif args.dataset == 'Clothing1M':
        trainset = Clothing1M(root=args.root_dir, mode='dirty_train', transform=transform_train)
        trainset_track = Clothing1M(root=args.root_dir, mode='dirty_train', transform=transform_train)
        testset = Clothing1M(root=args.root_dir, mode='test', transform=transform_test)
        trainset_clean_transform = Clothing1M(root=args.root_dir, mode='dirty_train', transform=transform_clean)
        valset_clean_transform = Clothing1M(root=args.root_dir, mode='val', transform=transform_clean)
        num_classes = 14
        use_val_set = True  # Probes are defined on the validation set
    else:
        raise NotImplementedError
    ssl_training = args.ssl_training
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.dataset == "Clothing1M":
        print("Using ResNet-50...")
        model = models.resnet50(num_classes=num_classes).to(device)
        model = ModelWithFeatures(model)  # Returns model features as well
    else:
        model = PreResNet.ResNet18(num_classes=num_classes, ssl_training=ssl_training).to(device)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        print(f"Using SGD as the optimizer w/ LR={args.lr} / Momentum: {args.momentum} / Weight-decay: {args.weight_decay}")
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using Adam as the optimizer w/ LR={args.lr} / Weight-decay: {args.weight_decay}")
    else:
        assert args.optimizer == "adamw"
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW as the optimizer w/ LR={args.lr} / Weight-decay: {args.weight_decay}")
    
    scheduler = None
    if not args.no_milestones:
        milestones = args.M
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        print("Using LR scheduler with milestones:", milestones)

    noised_input_idx = None
    post_proc_transform = None
    available_indices = None
    
    if args.dataset == "Clothing1M":
        assert args.noise_level == 0.0, f"Noise level should be set to zero for clothing1M dataset (provided noise level={args.noise_level})"
        available_indices = [i for i in range(len(train_loader.dataset))]
        print("!! Not adding noise for clothing1M dataset...")
        
        # path where experiments are saved
        exp_path = os.path.join('./', 'prioritized_training_PreResNet18_{0}'.format(args.experiment_name))
    else:
        if args.use_three_sets:
            # Ensure that the labels change and the input noise is mutually exclusive
            print("!! Generating three different sets in the dataset...")
            # assert post_proc_transform is not None
            assert args.noise_level < 50.
            noised_input_idx = add_input_noise_cifar_w(train_loader, args.noise_level, post_proc_transform=None)  # it changes the labels in the train loader directly
            _ = add_input_noise_cifar_w(train_loader_track, args.noise_level, post_proc_transform=None)  # it changes the labels in the train loader directly
            
            labels = get_data_cifar_2(train_loader_track)  # it should be "clonning"
            noisy_labels, noised_label_idx = add_noise_cifar_w_new(train_loader, args.noise_level, noised_input_idx)  # it changes the labels in the train loader directly
            noisy_labels_track, noised_label_idx_track = add_noise_cifar_w_new(train_loader_track, args.noise_level, noised_input_idx)
            
            available_indices = [i for i in range(len(labels)) if i not in noised_input_idx and i not in noised_label_idx]
            print("Number of available indices:", len(available_indices))
            
            # Three sets have to be: corrupted -> noisy ; noisy -> mislabaled
            args.use_mislabeled_examples = True
        
        else:
            print("!! Using the default label noise pipeline...")
            assert post_proc_transform is None
            labels = get_data_cifar_2(train_loader_track)  # it should be "clonning"
            noisy_labels = add_noise_cifar_w(train_loader, args.noise_level)  # it changes the labels in the train loader directly
            noisy_labels_track = add_noise_cifar_w(train_loader_track, args.noise_level)
            
            # Unchanged indices
            # TODO: Further filter the scores based on which examples are available
            available_indices = [i for i in range(len(labels)) if labels[i] == noisy_labels[i]]
    
        noisy_labels = torch.Tensor(noisy_labels)
        misclassified_instances = labels != noisy_labels
        print(f"Percentage of changed instances: {torch.sum(misclassified_instances)/float(len(misclassified_instances))*100.:2f}% ({torch.sum(misclassified_instances)}/{len(misclassified_instances)}) / Noise: {args.noise_level}")
        
        # path where experiments are saved
        exp_path = os.path.join('./', 'noise_models_PreResNet18_{0}'.format(args.experiment_name), str(args.noise_level))
    
    assert not args.dynamic_flood_thresh or args.flood_test
    if args.flood_test:
        assert args.reg_term == 0.
        assert args.BootBeta == "None"
        assert args.Mixup in ["None", "Flooding"]
        print("Using flooding test...")

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
            if not args.resume_from_pretraining:
                # Remove the old directory and recreate it
                shutil.rmtree(exp_path)
                os.makedirs(exp_path)
                print("Recreated the output directory after deleting old results...")
    bmm_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0

    if not args.no_milestones:
        bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
        guidedMixup_ep = 106

    if args.Mixup == 'Dynamic':
        assert not args.no_milestones
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        if args.bootstrap_epochs is not None:
            bootstrap_ep_mixup = int(args.bootstrap_epochs) + 1
        else:
            assert not args.no_milestones
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
        if args.dataset == "Clothing1M":
            tensor_shape = (3, 224, 224)
        else:
            tensor_shape = (3, 32, 32)  # For both CIFAR-10/100
        if args.num_example_probes is not None:
            num_example_probes = args.num_example_probes
        else:
            num_example_probes = 250  # 0.5% of the dataset
        print("Selected number of example probes:", num_example_probes)
        normalizer = transforms.Normalize(mean, std)
        
        indices_to_remove = []
        if args.use_mislabeled_examples:
            print("Using examples from the dataset with random labels as probe...")
            
            assert available_indices is not None
            assert len(available_indices) > num_example_probes, f"Example probes: {num_example_probes} / Available indices: {len(available_indices)}"
            selected_indices = np.random.choice(available_indices, size=num_example_probes, replace=False)
            assert len(np.unique(selected_indices)) == len(selected_indices)
            
            if args.dataset == "Clothing1M":
                if use_val_set:
                    images = [valset_clean_transform[i][0] for i in selected_indices]  # Will include clean augmentations
                else:
                    images = [trainset_clean_transform[i][0] for i in selected_indices]  # Will include clean augmentations
                probes["noisy"] = torch.stack(images, dim=0)
            else:
                assert not use_val_set
                transforms_clean = transforms.ToTensor()
                images = [train_loader.sampler.data_source.data[i] for i in selected_indices]
                probes["noisy"] = torch.stack([transforms_clean(x) for x in images], dim=0)
            print("Selected image shape:", probes["noisy"].shape)
            
            # Update available indices
            available_indices = [i for i in available_indices if i not in selected_indices]
            indices_to_remove += [i for i in selected_indices]
        
        else:
            if args.treat_three_sets or args.use_gmm_probe_identification:
                raise NotImplementedError("Definition of corrupted probe is ambiguous...")
            print("Creating random examples with random labels as probe...")
            probes["noisy"] = torch.empty(num_example_probes, *tensor_shape).uniform_(0., 1.)
        
        probe_list = ["noisy"]
        random_gen_labels = probe_list
        if args.treat_three_sets or args.use_gmm_probe_identification:
            assert args.use_mislabeled_examples, "Three sets only supports mislabeled probes for mislabeled example detection"
            assert args.BootBeta == "HardProbes", "Only HardProbes is supported for three-set treatment"
            
            # Include another probe for the detection of the third set
            assert available_indices is not None
            assert len(available_indices) > num_example_probes, f"Example probes: {num_example_probes} / Available indices: {len(available_indices)}"
            use_random_inputs = True  # Mislabeled examples are detected using mislabeled probe
            if use_random_inputs:
                probes["corrupted"] = torch.empty(num_example_probes, *tensor_shape).uniform_(0., 1.)
            else:
                raise NotImplementedError
                selected_indices = np.random.choice(available_indices, size=(num_example_probes,), replace=False)
                # TODO: Random labels cannot be used in this case...
            
            probe_list = ["noisy", "corrupted", "typical"] if args.treat_three_sets else ["typical", "noisy"]
            random_gen_labels = ["noisy", "corrupted"]
            print("Corrupted probe included in the dataset for three-set treatment...")
            
            print("Using examples from the dataset with original labels as typical probe...")
            selected_indices = np.random.choice(available_indices, size=num_example_probes, replace=False)
            assert len(np.unique(selected_indices)) == len(selected_indices)
            
            if args.dataset == "Clothing1M":
                if use_val_set:
                    images = [valset_clean_transform[i] for i in selected_indices]  # Will include clean augmentations
                else:
                    images = [trainset_clean_transform[i] for i in selected_indices]  # Will include clean augmentations
                probes["typical"] = torch.stack([x[0] for x in images], dim=0)
                probes[f"typical_labels"] = torch.tensor([x[1] for x in images], dtype=torch.int64).to(device)
            else:
                assert not use_val_set
                images = [train_loader.sampler.data_source.data[i] for i in selected_indices]
                probes["typical"] = torch.stack([transforms_clean(x) for x in images], dim=0).to(device)
                probes[f"typical_labels"] = torch.tensor([train_loader.sampler.data_source.targets[i] for i in selected_indices], dtype=torch.int64).to(device)
            print("Selected image shape:", probes["typical"].shape)
            
            indices_to_remove += [i for i in selected_indices]
        
        if len(indices_to_remove) > 0 and not use_val_set:  # If val set is used, don't discard train set samples
            # Remove these examples from the dataset
            print(f"Dataset before deletion: {len(trainset)} / Dataloader size: {len(train_loader)}")
            print("Number of indices to be removed:", len(indices_to_remove))
            num_total_examples = len(trainset)
            if args.dataset == "Clothing1M":
                trainset.imgs = [trainset.imgs[i] for i in range(num_total_examples) if i not in indices_to_remove]
                trainset.targets = [trainset.targets[i] for i in range(num_total_examples) if i not in indices_to_remove]
                misclassified_instances = [False for i in range(num_total_examples) if i not in indices_to_remove]  # Don't have this explicit information
            else:
                trainset.data = [trainset.data[i] for i in range(num_total_examples) if i not in indices_to_remove]
                trainset.targets = [trainset.targets[i] for i in range(num_total_examples) if i not in indices_to_remove]
                misclassified_instances = [misclassified_instances[i] for i in range(num_total_examples) if i not in indices_to_remove]
            
            # Reinitialize the dataloader to generate the right indices for sampler
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
            print(f"Dataset after deletion: {len(trainset)} / Dataloader size: {len(train_loader)}")
        
        assert all([probes[k].shape == (num_example_probes, *tensor_shape) for k in probe_list])
        for k in probe_list:
            probes[k] = normalizer(probes[k]).to(device)
            if k in random_gen_labels:
                print("Generating random labels for:", k)
                probes[f"{k}_labels"] = torch.randint(0, num_classes, (num_example_probes,)).to(device)
        
        probe_images = torch.cat([probes[k] for k in probe_list], dim=0)
        probe_labels = torch.cat([probes[f"{k}_labels"] for k in probe_list], dim=0)
        probe_dataset_standard = CustomTensorDataset(probe_images.to("cpu"), [int(x) for x in probe_labels.to("cpu").numpy().tolist()])
        if use_val_set:
            print("! WARNING: Not including probe examples for training taken from the validation set...")
            comb_trainset = trainset
            dataset_probe_identity = ["train_noisy" if misclassified_instances[i] else "train_clean" for i in range(len(trainset))]
        else:
            comb_trainset = torch.utils.data.ConcatDataset([trainset, probe_dataset_standard])
            
            # probe_identity = ["noisy_probe" for _ in range(len(probe_images))]
            probe_identity = [f"{k}_probe" for k in probe_list for _ in range(num_example_probes)]
            assert len(probe_identity) == len(probe_images)
            dataset_probe_identity = ["train_noisy" if misclassified_instances[i] else "train_clean" for i in range(len(trainset))] + probe_identity
        
        assert len(dataset_probe_identity) == len(comb_trainset)
        print("Probe dataset:", len(comb_trainset), comb_trainset[0][0].shape)
        
        idx_dataset = IdxDataset(comb_trainset, dataset_probe_identity)
        idx_train_loader = torch.utils.data.DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        train_loader_w_probes = torch.utils.data.DataLoader(comb_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        total_instances = len(idx_dataset)
        noisy_probe_instances = np.sum([1 if dataset_probe_identity[i] == "noisy_probe" else 0 for i in range(len(idx_dataset))])
        typical_probe_instances = np.sum([1 if dataset_probe_identity[i] == "typical_probe" else 0 for i in range(len(idx_dataset))])
        corrupted_probe_instances = np.sum([1 if dataset_probe_identity[i] == "corrupted_probe" else 0 for i in range(len(idx_dataset))])
        noisy_train_instances = np.sum([1 if dataset_probe_identity[i] == "train_noisy" else 0 for i in range(len(idx_dataset))])
        clean_train_instances = np.sum([1 if dataset_probe_identity[i] == "train_clean" else 0 for i in range(len(idx_dataset))])
        print(f"Total instances: {total_instances} / Typical probe instances: {typical_probe_instances} / Corrupted probe instances: {corrupted_probe_instances} / Noisy probe instances: {noisy_probe_instances} / Noisy train instances: {noisy_train_instances} / Clean train instances: {clean_train_instances}")
    else:
        idx_dataset = IdxDataset(trainset, None)
        idx_train_loader = torch.utils.data.DataLoader(idx_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    probe_detection_list = []
    bmm_detection_list = []
    prob_model = None
    trajectory_set = {"typical": [], "corrupted": [], "noisy": [], "train": []}
    model_loaded = False

    for epoch in range(1, args.epochs + 1):
        # train
        if scheduler is not None:
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
                if args.selection_batch_size is not None:
                    # Executes the new loss trajectory functions without mixup for prioritized training
                    if args.BootBeta == "None":
                        if args.use_loss_trajectories:
                            print('\t##### Doing CE loss-based training with loss trajectories (store for identification) #####')
                            trajectory_set = train_CrossEntropy_traj(args, model, device, idx_train_loader, optimizer, epoch, trajectory_set)
                            assert probes is not None
                            
                            msg = f"Probe during pretraining{' (train set + typical probe)' if args.use_probes_for_pretraining else ''}"
                            typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg=msg)
                            if "corrupted" in probes and len(probes["corrupted"]) > 0 and "corrupted_labels" in probes:
                                msg = f"Probe during pretraining{' (train set + corrupted probe)' if args.use_probes_for_pretraining else ''}"
                                corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"], msg="Corrupted probe")
                                trajectory_set["corrupted"].append(corrupted_stats["loss_vals"])
                            msg = f"Probe during pretraining{' (train set + noisy probe)' if args.use_probes_for_pretraining else ''}"
                            noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg=msg)
                            trajectory_set["typical"].append(typical_stats["loss_vals"])
                            trajectory_set["noisy"].append(noisy_stats["loss_vals"])
                            
                            # Save the trajectories in the last epoch
                            print(f"Saving the loss trajectories after epoch # {epoch}...")
                            traj_output_file = os.path.join(exp_path, f'{args.dataset.lower()}_loss_trajectories_ep_{epoch}.pkl')
                            with open(traj_output_file, 'wb') as fp:
                                pickle.dump(trajectory_set, fp, protocol=pickle.HIGHEST_PROTOCOL)
                                print("Data saved to output file:", traj_output_file)
                        else:
                            print(f'\t##### Doing CE loss-based training with uniform online batch selection ({args.selection_batch_size} / {args.batch_size}) #####')
                            trajectory_set = train_CrossEntropy_traj(args, model, device, idx_train_loader, optimizer, epoch, trajectory_set, 
                                                                     selection_batch_size=args.selection_batch_size)
                    else:
                        print(f'\t##### Doing CE loss-based training with probe-based online batch selection ({args.selection_batch_size} / {args.batch_size}) #####')
                        
                        assert args.BootBeta == "HardProbes"
                        if epoch == 1:  # Load only in the first epoch
                            print("Loading trajectory set from file:", args.loss_trajectories_path)
                            with open(traj_output_file, 'rb') as fp:
                                trajectory_set = pickle.load(fp)
                                print("Trajectories loaded successfully...")
                                typical_traj = np.array(trajectory_set["typical"]).transpose(1, 0)
                                noisy_traj = np.array(trajectory_set["noisy"]).transpose(1, 0)
                                train_traj = np.array(trajectory_set["train"]).transpose(1, 0)
                                print(f"Trajectories shape / Train: {train_traj.shape} / Typical: {typical_traj.shape} / Noisy: {noisy_traj.shape}")
                        
                        # Execute the label correction scheme
                        train_CrossEntropy_loss_traj_prioritized_typical(args, model, device, idx_train_loader, optimizer, epoch, args.reg_term, 
                                                                         num_classes, probes, trajectory_set, not args.use_binary_prediction,
                                                                         selection_batch_size=args.selection_batch_size)
                
                else:
                    print('\t##### Doing standard training with cross-entropy loss #####')
                    loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, use_ssl=ssl_training)

        ### Mixup ###
        if args.Mixup == "Static":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                if args.resume_from_pretraining:
                    continue  # Skip the step
                
                if args.use_loss_trajectories:
                    print('\t##### Doing NORMAL mixup with loss trajectories for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                    trajectory_set = train_mixUp_traj(args, model, device, idx_train_loader, optimizer, epoch, 32, trajectory_set)
                    assert probes is not None
                    
                    msg = f"Probe during pretraining{' (train set + typical probe)' if args.use_probes_for_pretraining else ''}"
                    typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg=msg)
                    if "corrupted" in probes and len(probes["corrupted"]) > 0 and "corrupted_labels" in probes:
                        msg = f"Probe during pretraining{' (train set + corrupted probe)' if args.use_probes_for_pretraining else ''}"
                        corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"], msg="Corrupted probe")
                        trajectory_set["corrupted"].append(corrupted_stats["loss_vals"])
                    msg = f"Probe during pretraining{' (train set + noisy probe)' if args.use_probes_for_pretraining else ''}"
                    noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg=msg)
                    trajectory_set["typical"].append(typical_stats["loss_vals"])
                    trajectory_set["noisy"].append(noisy_stats["loss_vals"])
                    
                    if args.bootstrap_probe_acc_thresh is not None:
                        if noisy_stats["acc"] >= args.bootstrap_probe_acc_thresh:
                            print(f"!! Accuracy on probe exceeded to {noisy_stats['acc']}% (threhold={args.bootstrap_probe_acc_thresh}%). Stopping pretraining...")
                            bootstrap_ep_mixup = epoch
                elif args.use_unmodified_train_set_for_pretraining:
                    print('\t##### Doing NORMAL mixup on unmodified train set for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader_track, optimizer, epoch, 32)
                else:
                    print('\t##### Doing NORMAL mixup{0} for {1} epochs #####'.format(' with probes' if args.use_probes_for_pretraining else '', bootstrap_ep_mixup - 1))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader_w_probes if args.use_probes_for_pretraining else train_loader, optimizer, epoch, 32)
                    if probes is not None:
                        # Evaluate the model performance on
                        if args.use_gmm_probe_identification:
                            msg = f"Probe during pretraining{' (train set + typical probe)' if args.use_probes_for_pretraining else ''}"
                            typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg=msg)
                        msg = f"Probe during pretraining{' (train set + noisy probe)' if args.use_probes_for_pretraining else ''}"
                        noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg=msg)
                    
                    if args.bootstrap_probe_acc_thresh is not None:
                        if noisy_stats["acc"] >= args.bootstrap_probe_acc_thresh:
                            print(f"!! Accuracy on probe exceeded to {noisy_stats['acc']}% (threhold={args.bootstrap_probe_acc_thresh}%). Stopping pretraining...")
                            bootstrap_ep_mixup = epoch

            else:
                pretrained_snap = f"pretrained_model{'_probes' if args.use_probes_for_pretraining else ''}"
                model_checkpoint = os.path.join(exp_path, pretrained_snap + '.pth')
                optimizer_checkpoint = os.path.join(exp_path, 'opt_' + pretrained_snap + '.pth')
                
                if args.resume_from_pretraining and not model_loaded:
                    assert os.path.exists(model_checkpoint)
                    assert os.path.exists(optimizer_checkpoint)
                    print("Loading pretrained checkpoint...")
                    model.load_state_dict(torch.load(model_checkpoint))
                    optimizer.load_state_dict(torch.load(optimizer_checkpoint))
                    
                    # Determine the best test accuracy of the pretrained model
                    loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)
                    best_acc_val = acc_val_per_epoch_i[-1]
                    snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                        epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
                    
                    # Recreate the BMM model
                    epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                        track_training_loss(args, model, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
                    
                    model_loaded = True
                    print(f"Pretrained model accuracy: {best_acc_val}%")
                else:
                    # Save the pretrained checkpoint
                    if not os.path.exists(model_checkpoint):
                        print("Saving the checkpoint file here after the pretraining phase...")
                        torch.save(model.state_dict(), model_checkpoint)
                        torch.save(optimizer.state_dict(), optimizer_checkpoint)
                
                if args.BootBeta == "Hard":
                    print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,\
                                                                                     alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)
                if args.BootBeta == "HardProbes":
                    if args.use_loss_trajectories:
                        if args.treat_three_sets:
                            print("\t##### Doing HARD BETA bootstrapping with loss trajectories using three sets and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                            loss_per_epoch, acc_train_per_epoch_i, prob_model = train_mixUp_HardBootBeta_probes_three_sets_loss_traj(args, model, device, idx_train_loader, optimizer, epoch,\
                                                                                            alpha, args.reg_term, num_classes, probes, trajectory_set, not args.use_binary_prediction)
                        else:
                            print("\t##### Doing HARD BETA bootstrapping with loss trajectories and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                            loss_per_epoch, acc_train_per_epoch_i, trajectory_set = train_mixUp_HardBootBeta_probes_loss_traj(args, model, device, idx_train_loader, optimizer, epoch,\
                                                                                            alpha, args.reg_term, num_classes, probes, trajectory_set, not args.use_binary_prediction)
                    elif args.treat_three_sets:
                        print("\t##### Doing HARD BETA bootstrapping with Probes using three sets and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                        loss_per_epoch, acc_train_per_epoch_i, prob_model = train_mixUp_HardBootBeta_probes_three_sets(args, model, device, train_loader_w_probes, optimizer, epoch,\
                                                                                        alpha, args.reg_term, num_classes, probes, prob_model, not args.use_bmm_treatment,
                                                                                        args.use_adaptive_weights, args.update_every_iter)
                    elif args.use_gmm_probe_identification:
                        print("\t##### Doing HARD BETA bootstrapping with GMM combined with Probes and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                        loss_per_epoch, acc_train_per_epoch_i, prob_model = train_mixUp_HardBootBeta_probes_gmm(args, model, device, train_loader_w_probes, optimizer, epoch,\
                                                                                        alpha, args.reg_term, num_classes, probes, prob_model, not args.use_bmm_treatment,
                                                                                        args.use_adaptive_weights, args.use_binary_prediction, args.use_softmax, args.update_every_iter)
                    else:
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
