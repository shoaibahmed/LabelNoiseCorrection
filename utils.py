from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt

import sklearn.neighbors
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing as preprocessing
import sys
from tqdm import tqdm

import os
import cv2
import kornia.augmentation

import scipy.stats
import scipy.special

from style_transfer import style_transfer
from stylized_cifar10.style_transfer_cifar import StyleTransfer


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


class ModelWithFeatures(torch.nn.Module):
    def __init__(self, model, use_projection_head=False, feat_dim=None):
        super().__init__()
        self.model = model
        self.fc_layer = model.fc
        self.model.fc = torch.nn.Identity()
        
        self.projection_head = None
        if use_projection_head:
            assert feat_dim is not None
            print("Using projection head with Supervised Contrastive loss...")
            self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, feat_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(feat_dim, feat_dim)
        )
    
    def forward(self, x, return_features=False):
        features = self.model(x)
        logits = self.fc_layer(features)
        if not return_features:
            return logits
        
        if self.projection_head is not None:
            features = self.projection_head(features)
        return features, logits


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return torch.clip(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0.0, 1.0)


######################### Get data and noise adding ##########################
def get_data_cifar(loader):
    data = loader.sampler.data_source.train_data.copy()
    labels = loader.sampler.data_source.targets
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return (data, labels)

def get_data_cifar_2(loader):
    labels = loader.sampler.data_source.targets
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return labels

#Noise without the sample class
def add_noise_cifar_wo(loader, noise_percentage = 20):
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(
                set(range(10)) - set([label_i]))  # this is a set with the available labels (without the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels

    return noisy_labels

#Noise with the sample class (as in Re-thinking generalization )
def add_noise_cifar_w(loader, noise_percentage = 20):
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels

    return noisy_labels

#Noise with the sample class (as in Re-thinking generalization )
def add_noise_cifar_w_new(loader, noise_percentage = 20, changed_idx_list: list = None):
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    available_idx = [i for i in range(len(loader.sampler.data_source.targets)) if i not in changed_idx_list]
    
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    noisy_input_percentage = 100. * len(changed_idx_list) / len(noisy_labels)
    idx_to_change = probs_to_change >= (100.0 - noise_percentage - noisy_input_percentage)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))

    changed_idx = []
    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1 and n not in changed_idx_list:
            set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]
            changed_idx.append(n)

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels
    print(f"Total examples: {len(noisy_labels)} / Noisy inputs: {len(changed_idx_list)} / Total labels changed: {len(changed_idx)} ")

    assert all([i not in changed_idx_list for i in changed_idx])
    return noisy_labels, changed_idx

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class ClampRangeTransform(object):
    def __init__(self, min_range, max_range):
        self.min_range = min_range
        self.max_range = max_range
    
    def __call__(self, x):
        return torch.clamp(x, self.min_range, self.max_range)

def add_input_noise(x):
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 3
    assert x.shape == (3, 32, 32)
    assert x.dtype == torch.uint8

    max_val = x.abs().max()
    # print("Max val in image:", max_val)
    noise_std = max_val * 0.1  # 10% of the maximum deviation
    
    # Color jitter after normalization?
    corruption_transform = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                                               transforms.RandomApply([AddGaussianNoise(mean=0.0, std=noise_std)], p=0.5),
                                               transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0))], p=0.5),
                                               ClampRangeTransform(0, 255)])
    
    x = corruption_transform(x)
    return x.to(torch.uint8)

def add_input_noise_np(x):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 3
    assert x.shape == (32, 32, 3)

    x = torch.from_numpy(x).permute(2, 0, 1)  # HWC -> CHW
    x = add_input_noise(x)
    x = x.permute(1, 2, 0).numpy()  # CHW -> HWC
    
    assert x.shape == (32, 32, 3)
    return x

# Add input noise to the examples
def add_input_noise_cifar_w(loader, noise_percentage=20, post_proc_transform=None, 
                            use_style_transfer=False, use_edge_detection=False, use_random_inputs=True):
    assert post_proc_transform is None
    assert not (use_style_transfer and use_edge_detection)
    
    torch.manual_seed(2)
    np.random.seed(42)
    noisy_labels = [sample_i for sample_i in loader.sampler.data_source.targets]
    images = [sample_i for sample_i in loader.sampler.data_source.data]
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)
    
    write_num_outputs = 5
    iterator = 0
    style_transfer_cls = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_style_transfer:
        print("Using style transfer to generate input noise...")
        style_root_dir = "./style_imgs/"
        style_img_list = os.listdir(style_root_dir)
        print("Total style images found:", len(style_img_list))
        style_img_list = [os.path.join(style_root_dir, x) for x in style_img_list]
        # style_img_list = [cv2.resize(cv2.imread(x), (224, 224)) for x in style_img_list]
        style_img_list = [cv2.resize(cv2.imread(x), (32, 32)) for x in style_img_list]
        style_transfer_cls = StyleTransfer(device)
    
    changed_idx = []
    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            assert isinstance(images[n], np.ndarray)
            assert images[n].dtype == np.uint8
            if iterator < write_num_outputs:
                cv2.imwrite(f"before_noise_{iterator}.png", cv2.resize(images[n], (224, 224)))
            
            # Augment the np array
            if use_style_transfer:
                style_img_idx = np.random.randint(len(style_img_list))
                out = style_transfer_cls.style_transfer(style_img_list[style_img_idx], images[n])
                # out = style_transfer(cv2.resize(images[n], (224, 224)), style_img_list[style_img_idx], device=device)
                # style_img_idx = np.random.randint(len(images))
                # out = style_transfer_cls.style_transfer(images[style_img_idx], images[n])
                images[n] = out
            elif use_edge_detection:
                img = cv2.GaussianBlur(images[n], (1, 1), 0)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # out = cv2.Canny(img, 20, 50)
                out = cv2.Laplacian(img, cv2.CV_64F)
                out = cv2.convertScaleAbs(out)
                images[n] = np.stack([out, out, out], axis=2)
            elif use_random_inputs:
                random_img = np.clip(np.random.rand(32, 32, 3) * 255, 0, 255).astype(np.uint8)
                images[n] = random_img
            else:
                images[n] = add_input_noise_np(images[n])
            
            assert isinstance(images[n], np.ndarray)
            assert images[n].dtype == np.uint8
            assert images[n].shape == (32, 32, 3)
            if iterator < write_num_outputs:
                cv2.imwrite(f"after_noise_{iterator}.png", cv2.resize(images[n], (224, 224)))
                iterator += 1
            
            changed_idx.append(n)
        
        if post_proc_transform is not None:
            images[n] = post_proc_transform(images[n])

    loader.sampler.data_source.data = images
    loader.sampler.data_source.targets = noisy_labels
    
    return changed_idx


##############################################################################


##################### Loss tracking and noise modeling #######################


def track_training_loss(args, model, device, train_loader, epoch, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)

        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, target, reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
        probs = prediction.clone()
        probs.detach_()
        all_probs = torch.cat((all_probs, probs.cpu()))
        arg_entr = torch.max(prediction, dim=1)[1]
        arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction='none')
        arg_entr.detach_()
        all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))

    loss_tr = all_losses.data.numpy()

    # outliers detection
    max_perc = np.percentile(loss_tr, 95)
    min_perc = np.percentile(loss_tr, 5)
    loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

    bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
    bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6


    loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

    loss_tr[loss_tr>=1] = 1-10e-4
    loss_tr[loss_tr <= 0] = 10e-4

    bmm_model = BetaMixture1D(max_iters=10)
    bmm_model.fit(loss_tr)
    print(bmm_model)

    bmm_model.create_lookup(1)

    return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_maxLoss, bmm_model_minLoss

##############################################################################

########################### Cross-entropy loss ###############################

def feature_loss(features_a, features_b):
    criterion = nn.MSELoss()
    return criterion(features_a, features_b)


def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, use_ssl=False, use_mse=True):
    model.train()
    loss_per_batch = []
    
    ssl_criterion = nn.CrossEntropyLoss()
    ssl_lambda = 0.1
    
    augmentation_func = None
    if use_ssl:
        input_dim = 32
        augmentation_list = [# kornia.augmentation.RandomRotation(degrees=(-45.0, 45.0)),
                             # kornia.augmentation.RandomPerspective(p=0.5, distortion_scale=0.25),
                             kornia.augmentation.RandomResizedCrop((input_dim, input_dim), scale=(0.75, 0.75)),
                             kornia.augmentation.ColorJitter(brightness=0.1, hue=0.1, saturation=0.1, contrast=0.1)]
        augmentation_func = nn.Sequential(*augmentation_list).to(device)
        print(f"Using auxillary SSL objective{' with MSE' if use_mse else ''} with regular CE loss...")
    
    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        features, output = model(data, return_features=True)
        output = F.log_softmax(output, dim=1)

        loss = F.nll_loss(output, target)
        
        if use_ssl:
            data_aug = augmentation_func(data)
            features_aug, _ = model(data_aug, return_features=True)
            if use_mse:
                ssl_loss = feature_loss(features, features_aug)
            else:
                ssl_logits, ssl_labels = info_nce_loss(features, features_aug)
                ssl_loss = ssl_criterion(ssl_logits, ssl_labels)
            
            # Average the loss from the two views
            # batch_size = len(data)
            # ssl_loss = (ssl_loss[:batch_size] + ssl_loss[batch_size:]) / 2.
            loss = loss + ssl_lambda * ssl_loss

        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


def train_CrossEntropy_traj(args, model, device, train_loader, optimizer, epoch, trajectory_set, selection_batch_size=None):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    total = 0
    
    example_idx = []
    loss_vals = []
    
    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        if selection_batch_size is not None:  # Uniform sample selection
            assert isinstance(selection_batch_size, int)
            selected_indices = torch.randperm(len(data))[:selection_batch_size]
            data, target = data[selected_indices], target[selected_indices]
        
        output = model(data, return_features=False)
        output = F.log_softmax(output, dim=1)

        example_loss = F.nll_loss(output, target, reduction='none')
        loss = example_loss.mean()
        
        # Compute individual example losses
        with torch.no_grad():
            example_idx.append(ex_idx.clone().cpu())
            loss_vals.append(example_loss.clone().cpu())

        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        acc_train_per_batch.append(100. * correct / total)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, # examples: {:d}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / total,
                optimizer.param_groups[0]['lr'], len(data)))

    if selection_batch_size is None:  # Some samples are empty otherwise
        example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
        loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
        
        # Sort the loss list
        sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
        for i in range(len(example_idx)):
            assert sorted_loss_list[example_idx[i]] is None
            sorted_loss_list[example_idx[i]] = loss_vals[i]
        assert not any([x is None for x in sorted_loss_list])
        
        # Append the loss list to loss trajectory
        if trajectory_set is None:
            trajectory_set = dict(train=[sorted_loss_list])
        else:
            assert "train" in trajectory_set
            trajectory_set["train"].append(sorted_loss_list)

        return trajectory_set


def train_CrossEntropy_loss_traj_prioritized_typical(args, model, device, train_loader, optimizer, epoch,
                                                     reg_term, num_classes, probes, trajectory_set, use_probs,
                                                     selection_batch_size=None):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    total = 0
    
    example_idx = []
    loss_vals = []
    
    typical_trajectories = np.array(trajectory_set["typical"]).transpose(1, 0)
    noisy_trajectories = np.array(trajectory_set["noisy"]).transpose(1, 0)
    train_trajectories = np.array(trajectory_set["train"]).transpose(1, 0)
    print(f"Typical trajectory size: {typical_trajectories.shape} / Noisy trajectories shape: {noisy_trajectories.shape}")
    print(f"Train trajectories shape: {train_trajectories.shape}")
    
    probe_trajectories = np.concatenate([typical_trajectories, noisy_trajectories], axis=0)
    targets = np.array([0 for _ in range(len(typical_trajectories))] + [1 for _ in range(len(noisy_trajectories))])
    print(f"Combined probe trajectories: {probe_trajectories.shape} / Targets: {targets.shape}")
    
    n_neighbors = 20
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(probe_trajectories, targets)

    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        ex_trajs = np.array([train_trajectories[int(i)] for i in ex_idx])
        if selection_batch_size is not None:  # Typical score-based sample selection
            assert use_probs
            assert isinstance(selection_batch_size, int)
            B = clf.predict_proba(ex_trajs)  # 0 means typical and 1 means noisy
            assert len(B.shape) == 2 and B.shape[1] == 2, B.shape
            B = torch.from_numpy(np.array(B)).to(device)
            
            class_scores = B.mean(dim=0)
            print(f"Class scores / Typical: {class_scores[0]:.4f} / Noisy: {class_scores[1]:.4f}")
            B = B[:, 0]  # Only take the prob for being typical
            
            # Select examples with the highest probablity of being typical
            selected_indices = torch.argsort(B, descending=True)[:selection_batch_size]
            data, target = data[selected_indices], target[selected_indices]
        else:
            if use_probs:
                B = clf.predict_proba(ex_trajs)  # 0 means typical and 1 means noisy
                assert len(B.shape) == 2 and B.shape[1] == 2, B.shape
                B = B[:, 1]  # Only take the prob for being noisy
            else:
                B = clf.predict(ex_trajs)  # 1 means noisy
            B = torch.from_numpy(np.array(B)).to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        output = model(data, return_features=False)
        output = F.log_softmax(output, dim=1)
        pred = torch.max(output, dim=1)[1]

        if selection_batch_size is None:
            # Compute individual example losses
            with torch.no_grad():
                example_loss = F.nll_loss(output, target, reduction='none')
                example_idx.append(ex_idx.clone().cpu())
                loss_vals.append(example_loss.clone().cpu())

            loss_target_vec = (1 - B) * F.nll_loss(output, target, reduction='none')
            loss_target = torch.sum(loss_target_vec) / len(loss_target_vec)

            loss_pred_vec = B * F.nll_loss(output, pred, reduction='none')
            loss_pred = torch.sum(loss_pred_vec) / len(loss_pred_vec)

            loss = loss_target + loss_pred

            # loss_reg = reg_loss_class(tab_mean_class, num_classes)
            # loss = loss + reg_term*loss_reg
        else:
            loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        acc_train_per_batch.append(100. * correct / total)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, # examples: {:d}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / total,
                optimizer.param_groups[0]['lr'], len(data)))

    if selection_batch_size is None:
        example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
        loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
        
        # Sort the loss list
        sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
        for i in range(len(example_idx)):
            assert sorted_loss_list[example_idx[i]] is None
            sorted_loss_list[example_idx[i]] = loss_vals[i]
        assert not any([x is None for x in sorted_loss_list])
        
        # Append the loss list to loss trajectory
        if trajectory_set is None:
            trajectory_set = dict(train=[sorted_loss_list])
        else:
            assert "train" in trajectory_set
            trajectory_set["train"].append(sorted_loss_list)

        typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe")
        noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
        trajectory_set["typical"].append(typical_stats["loss_vals"])
        trajectory_set["noisy"].append(noisy_stats["loss_vals"])

        loss_per_epoch = [np.average(loss_per_batch)]
        acc_train_per_epoch = [np.average(acc_train_per_batch)]
        return (loss_per_epoch, acc_train_per_epoch, trajectory_set)


def train_CrossEntropy_loss_traj_prioritized_typical_rho(args, model, device, train_loader, optimizer, epoch,
                                                         reg_term, num_classes, probes, trajectory_set, use_probs,
                                                         selection_batch_size=None):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    total = 0
    
    example_idx = []
    loss_vals = []
    
    typical_trajectories = np.array(trajectory_set["typical"]).transpose(1, 0)
    noisy_trajectories = np.array(trajectory_set["noisy"]).transpose(1, 0)
    train_trajectories = np.array(trajectory_set["train"]).transpose(1, 0)
    print(f"Typical trajectory size: {typical_trajectories.shape} / Noisy trajectories shape: {noisy_trajectories.shape}")
    print(f"Train trajectories shape: {train_trajectories.shape}")
    
    probe_trajectories = np.concatenate([typical_trajectories, noisy_trajectories], axis=0)
    targets = np.array([0 for _ in range(len(typical_trajectories))] + [1 for _ in range(len(noisy_trajectories))])
    print(f"Combined probe trajectories: {probe_trajectories.shape} / Targets: {targets.shape}")
    
    n_neighbors = 20
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(probe_trajectories, targets)

    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        ex_trajs = np.array([train_trajectories[int(i)] for i in ex_idx])
        if selection_batch_size is not None:  # Typical score-based sample selection
            assert use_probs
            assert isinstance(selection_batch_size, int)
            B = clf.predict_proba(ex_trajs)  # 0 means typical and 1 means noisy
            assert len(B.shape) == 2 and B.shape[1] == 2, B.shape
            B = torch.from_numpy(np.array(B)).to(device)
            
            class_scores = B.mean(dim=0)
            B = B[:, 0]  # Only take the prob for being typical
            
            # Identify examples which are already learned (compute the prob)
            with torch.no_grad():
                output = model(data, return_features=False)
                output = F.softmax(output, dim=1)
                correct_class_probs = output[torch.arange(len(output)), target]
            typicality_score = B
            not_learned_score = 1. - correct_class_probs  # The higher the score, the more learned it is
            selection_score = (typicality_score + not_learned_score) / 2.
            print(f"Class scores / Typical: {class_scores[0]:.4f} / Noisy: {class_scores[1]:.4f} / Not learned score: {not_learned_score.mean():.4f} / Selection score: {selection_score.mean():.4f}")
            
            # Select examples with the highest probablity of being typical and lowest correct class prob
            selected_indices = torch.argsort(selection_score, descending=True)[:selection_batch_size]
            data, target = data[selected_indices], target[selected_indices]
        else:
            if use_probs:
                B = clf.predict_proba(ex_trajs)  # 0 means typical and 1 means noisy
                assert len(B.shape) == 2 and B.shape[1] == 2, B.shape
                B = B[:, 1]  # Only take the prob for being noisy
            else:
                B = clf.predict(ex_trajs)  # 1 means noisy
            B = torch.from_numpy(np.array(B)).to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        output = model(data, return_features=False)
        output = F.log_softmax(output, dim=1)
        pred = torch.max(output, dim=1)[1]

        if selection_batch_size is None:
            # Compute individual example losses
            with torch.no_grad():
                example_loss = F.nll_loss(output, target, reduction='none')
                example_idx.append(ex_idx.clone().cpu())
                loss_vals.append(example_loss.clone().cpu())

            loss_target_vec = (1 - B) * F.nll_loss(output, target, reduction='none')
            loss_target = torch.sum(loss_target_vec) / len(loss_target_vec)

            loss_pred_vec = B * F.nll_loss(output, pred, reduction='none')
            loss_pred = torch.sum(loss_pred_vec) / len(loss_pred_vec)

            loss = loss_target + loss_pred

            # loss_reg = reg_loss_class(tab_mean_class, num_classes)
            # loss = loss + reg_term*loss_reg
        else:
            loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        acc_train_per_batch.append(100. * correct / total)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, # examples: {:d}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / total,
                optimizer.param_groups[0]['lr'], len(data)))

    if selection_batch_size is None:
        example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
        loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
        
        # Sort the loss list
        sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
        for i in range(len(example_idx)):
            assert sorted_loss_list[example_idx[i]] is None
            sorted_loss_list[example_idx[i]] = loss_vals[i]
        assert not any([x is None for x in sorted_loss_list])
        
        # Append the loss list to loss trajectory
        if trajectory_set is None:
            trajectory_set = dict(train=[sorted_loss_list])
        else:
            assert "train" in trajectory_set
            trajectory_set["train"].append(sorted_loss_list)

        typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe")
        noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
        trajectory_set["typical"].append(typical_stats["loss_vals"])
        trajectory_set["noisy"].append(noisy_stats["loss_vals"])

        loss_per_epoch = [np.average(loss_per_batch)]
        acc_train_per_epoch = [np.average(acc_train_per_batch)]
        return (loss_per_epoch, acc_train_per_epoch, trajectory_set)


def train_CrossEntropy_loss_traj_prioritized_typical_rho_three_set(args, model, device, train_loader, optimizer, epoch,
                                                                   reg_term, num_classes, probes, trajectory_set, use_probs,
                                                                   selection_batch_size=None):
    """
    Should do online batch selection, as well as online batch updation with loss values.
    Only train on examples considered to the corrupted
    """
    assert selection_batch_size is not None
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    total = 0
    
    example_idx = []
    loss_vals = []
    recompute_iter = 5
    
    typical_trajectories = np.array(trajectory_set["typical"]).transpose(1, 0)
    corrupted_trajectories = np.array(trajectory_set["corrupted"]).transpose(1, 0)
    noisy_trajectories = np.array(trajectory_set["noisy"]).transpose(1, 0)
    train_trajectories = np.array(trajectory_set["train"]).transpose(1, 0)
    print(f"Typical trajectory size: {typical_trajectories.shape} / Noisy trajectories shape: {noisy_trajectories.shape}")
    print(f"Train trajectories shape: {train_trajectories.shape}")

    probe_trajectories = np.concatenate([typical_trajectories, corrupted_trajectories, noisy_trajectories], axis=0)
    targets = np.array([0 for _ in range(len(typical_trajectories))] + [1 for _ in range(len(corrupted_trajectories))] + [2 for _ in range(len(noisy_trajectories))])
    print(f"Combined probe trajectories: {probe_trajectories.shape} / Targets: {targets.shape}")
    
    n_neighbors = 20
    
    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Recompute the probe stats
        if batch_idx % recompute_iter == 0:
            typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe", model_train_mode=True)
            corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"], msg="Corrupted probe", model_train_mode=True)
            noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe", model_train_mode=True)

            # Concatenate with the probe trajectories
            all_loss_vals = np.concatenate([typical_stats["loss_vals"], corrupted_stats["loss_vals"], noisy_stats["loss_vals"]], axis=0)[:, None]  # N x 1
            aug_probe_trajectories = np.concatenate([probe_trajectories, all_loss_vals], axis=1)  # N x E + N x 1 = N x (E+1)
            print(f"Probe shapes / Probe traj: {probe_trajectories.shape} / Loss vals: {all_loss_vals.shape} / Aug probe traj: {aug_probe_trajectories.shape}")
            clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
            clf.fit(aug_probe_trajectories, targets)
        
        ex_trajs = np.array([train_trajectories[int(i)] for i in ex_idx])
        
        # Augment the loss trajectories with the current example loss
        with torch.no_grad():
            output = model(data, return_features=False)
            output = F.log_softmax(output, dim=1)
            example_loss = F.nll_loss(output, target, reduction='none').clone().cpu()
            example_idx.append(ex_idx.clone().cpu())
            loss_vals.append(example_loss)
            example_loss = example_loss.numpy()

            aug_ex_trajs = np.concatenate([ex_trajs, example_loss[:, None]], axis=1)  # Concatenate
            print(f"Example shapes / Ex traj: {ex_trajs.shape} / Loss vals: {example_loss.shape} / Aug ex traj: {aug_ex_trajs.shape}")
        
        assert use_probs
        assert isinstance(selection_batch_size, int)
        B = clf.predict_proba(aug_ex_trajs)  # 0 means typical and 1 means noisy
        assert len(B.shape) == 2 and B.shape[1] == 3, B.shape
        B = torch.from_numpy(np.array(B)).to(device)
        
        class_scores = B.mean(dim=0)
        # selection_score = B[:, 1]  # Only take the prob for being corrupted
        selection_score = B[:, 0]  # Only take the prob for being typical
        print(f"Class scores / Typical: {class_scores[0]:.4f} / Corrupted: {class_scores[1]:.4f} / Noisy: {class_scores[2]:.4f} / Selection score: {selection_score.mean():.4f}")
        # selection_score = torch.rand(len(data)).to(device)  # Uniform selection
        model.train()
        
        # Perform selection based on the selection score (select the highest scoring examples)
        selected_indices = torch.argsort(selection_score, descending=True)[:selection_batch_size]
        data, target = data[selected_indices], target[selected_indices]

        output = model(data, return_features=False)
        output = F.log_softmax(output, dim=1)
        pred = torch.max(output, dim=1)[1]
        
        loss = F.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        acc_train_per_batch.append(100. * correct / total)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, # examples: {:d}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / total,
                optimizer.param_groups[0]['lr'], len(data)))

    # Concatenate the computed scores to the final list
    example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
    loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
    
    # Sort the loss list
    sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
    for i in range(len(example_idx)):
        assert sorted_loss_list[example_idx[i]] is None
        sorted_loss_list[example_idx[i]] = loss_vals[i]
    assert not any([x is None for x in sorted_loss_list])
    
    # Append the loss list to loss trajectory
    if trajectory_set is None:
        trajectory_set = dict(train=[sorted_loss_list])
    else:
        assert "train" in trajectory_set
        trajectory_set["train"].append(sorted_loss_list)

    typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe")
    corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"], msg="Corrupted probe")
    noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
    trajectory_set["typical"].append(typical_stats["loss_vals"])
    trajectory_set["corrupted"].append(corrupted_stats["loss_vals"])
    trajectory_set["noisy"].append(noisy_stats["loss_vals"])

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch, trajectory_set)


##############################################################################

########################### Cross-entropy loss ###############################

def info_nce_loss(z1, z2, temperature=0.07):
    """
    Adapted from: https://github.com/sthalles/SimCLR/blob/master/simclr.py
    """
    # Concatenate the features from the two heads to form the actual features
    features = torch.cat([z1, z2], dim=0)
    n_views = 2
    device = z1.device
    batch_size = len(z1)
    
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = torch.nn.functional.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    assert similarity_matrix.shape == (n_views * batch_size, n_views * batch_size)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def train_CrossEntropy_probes(args, model, device, train_loader, optimizer, epoch, loss_thresh, 
                              use_thresh_as_flood=False, use_ex_weights=False, stop_learning=False, use_ssl=False):
    assert not stop_learning or use_ex_weights
    criterion = nn.CrossEntropyLoss(reduction='none')
    # ssl_criterion = nn.CrossEntropyLoss(reduction='none')
    ssl_criterion = nn.CrossEntropyLoss()
    ssl_lambda = 0.1
    
    augmentation_func = None
    if use_ssl:
        input_dim = 32
        augmentation_list = [# kornia.augmentation.RandomRotation(degrees=(-45.0, 45.0)),
                             # kornia.augmentation.RandomPerspective(p=0.5, distortion_scale=0.25),
                             kornia.augmentation.RandomResizedCrop((input_dim, input_dim), scale=(0.75, 0.75)),
                             kornia.augmentation.ColorJitter(brightness=0.1, hue=0.1, saturation=0.1, contrast=0.1)]
        augmentation_func = nn.Sequential(*augmentation_list).to(device)
        print("Using auxillary SSL objective...")
    
    model.train()
    loss_per_batch = []
    
    example_idx = []
    predictions = []
    targets = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        features, output = model(data, return_features=True)
        # output = F.log_softmax(output, dim=1)
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)

        ex_weights = torch.ones_like(loss) # / len(loss)  # Equal weight for averaging
        if loss_thresh is not None:
            if stop_learning:
                ex_weights[loss >= loss_thresh] = 0.  # Don't train on these examples
            else:
                with torch.no_grad():
                    if use_thresh_as_flood:
                        flooding_level = torch.empty_like(loss).fill_(loss_thresh)
                    else:
                        flooding_level = loss.clone().detach()
                    flooding_level[loss < loss_thresh] = 0.  # Remove the flooding barrier if the loss is below the threshold
                    ex_weights[loss >= loss_thresh] = ex_weights[loss >= loss_thresh] * 0.1  # Reduced weight for examples which are flooded
                loss = (loss - flooding_level).abs() + flooding_level
        assert loss.shape == (len(data),)
        if use_ex_weights:
            loss = (loss * ex_weights).sum() / ex_weights.sum()
        else:
            loss = loss.mean()
        
        if use_ssl:
            data_aug = augmentation_func(data)
            features_aug, _ = model(data_aug, return_features=True)
            ssl_logits, ssl_labels = info_nce_loss(features, features_aug)
            ssl_loss = ssl_criterion(ssl_logits, ssl_labels)
            
            # Average the loss from the two views
            # batch_size = len(data)
            # ssl_loss = (ssl_loss[:batch_size] + ssl_loss[batch_size:]) / 2.
            loss = loss + ssl_lambda * ssl_loss
        
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))
        
        predictions.append(output.argmax(dim=1).detach().cpu())
        example_idx.append(ex_idx.clone().cpu())
        targets.append(target.clone().cpu())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    
    example_idx = torch.cat(example_idx, dim=0)
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return (loss_per_epoch, acc_train_per_epoch), (example_idx, predictions, targets)


##############################################################################

############################# Mixup original #################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):

    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)

def train_mixUp(args, model, device, train_loader, optimizer, epoch, alpha):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)

        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        loss = mixup_criterion(output, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

##############################################################################

############################# Mixup with probes #################################
def train_mixUp_probes(args, model, device, train_loader, optimizer, epoch, alpha, loss_thresh, use_thresh_as_flood=False, use_ex_weights=False, stop_learning=False):
    assert not stop_learning or use_ex_weights
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.train()
    loss_per_batch = []
    
    example_idx = []
    predictions = []
    targets = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)

        # output = model(inputs)
        # output = F.log_softmax(output, dim=1)
        # loss = mixup_criterion(output, targets_a, targets_b, lam)
        
        output = model(data)
        loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)

        ex_weights = torch.ones_like(loss) / len(loss)  # Equal weight for averaging
        if loss_thresh is not None:
            if stop_learning:
                ex_weights[loss >= loss_thresh] = 0.  # Don't train on these examples
            else:
                with torch.no_grad():
                    if use_thresh_as_flood:
                        flooding_level = torch.empty_like(loss).fill_(loss_thresh)
                    else:
                        flooding_level = loss.clone().detach()
                    flooding_level[loss < loss_thresh] = 0.  # Remove the flooding barrier if the loss is below the threshold
                    ex_weights[loss >= loss_thresh] = ex_weights[loss >= loss_thresh] * 0.1  # Reduced weight for examples which are flooded
                loss = (loss - flooding_level).abs() + flooding_level
        assert loss.shape == (len(data),)
        if use_ex_weights:
            loss = (loss * ex_weights).sum()
        else:
            loss = loss.mean()  # Reduction has been disabled -- do explicit reduction
        
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))
        
        predictions.append(output.argmax(dim=1).detach().cpu())
        example_idx.append(ex_idx.clone().cpu())
        targets.append(target.clone().cpu())

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    
    example_idx = torch.cat(example_idx, dim=0)
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return (loss_per_epoch, acc_train_per_epoch), (example_idx, predictions, targets)

##############################################################################

########################## Mixup + Dynamic Hard Bootstrapping ##################################
# Mixup with hard bootstrapping using the beta model
def reg_loss_class(mean_tab,num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1./num_classes)*torch.log((1./num_classes)/items)
    return loss

def mixup_data_Boot(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, \
                            bmm_model_maxLoss, bmm_model_minLoss, reg_term, num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)

        B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)


        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)


        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

##############################################################################

def set_bn_train_mode(model, track_statistics):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d) or \
                isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
            m.train()
            m.track_running_stats = track_statistics


def test_tensor(model, data, target, msg=None, batch_size=None, model_train_mode=False):
    assert torch.is_tensor(data) and torch.is_tensor(target)
    assert len(data) == len(target), f"{len(data)} != {len(target)}"
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    if model_train_mode:
        # model.train()
        set_bn_train_mode(model, track_statistics=False)
    else:
        model.eval()
    with torch.no_grad():
        if batch_size is None or batch_size == len(data):
            output = model(data)
            loss_vals = criterion(output, target)
            test_loss = float(loss_vals.mean())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            total = len(data)
        else:
            assert batch_size is not None
            num_batches = int(np.ceil(len(data) / batch_size))
            correct, total = 0, 0
            test_loss = 0.
            loss_vals = []
            
            for i in range(num_batches):
                current_data = data[i*batch_size:(i+1)*batch_size]
                current_target = target[i*batch_size:(i+1)*batch_size]
                output = model(current_data)
                loss_vals.append(criterion(output, current_target))
                test_loss += float(loss_vals[-1].sum())
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(current_target.view_as(pred)).sum().item()
                total += len(current_data)
            
            test_loss = test_loss / total
            loss_vals = torch.cat(loss_vals, dim=0)
            assert total == len(data), f"{total} != {len(data)}"
    
    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total, 
                       loss_vals=loss_vals.detach().cpu().numpy())
    
    if model_train_mode:  # To reset track statistics var
        set_bn_train_mode(model, track_statistics=True)
    
    if msg is not None:
        print(f"{msg} | Mode: {'train' if model_train_mode else 'eval'} | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")
    
    return output_dict

def compute_is_noisy(data, target, model, probes, std_lambda=0.0, use_std=True):
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses.detach_()
        outputs.detach_()
        
        noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"])
        if use_std:
            loss_mean = np.mean(noisy_stats["loss_vals"])
            loss_std = np.std(noisy_stats["loss_vals"])
            acc = noisy_stats["acc"]
            assert len(noisy_stats["loss_vals"]) == len(probes["noisy_labels"]), f"{len(noisy_stats['loss_vals'])} != {len(probes['noisy_labels'])}"
            current_loss_thresh = max(loss_mean + std_lambda * loss_std, 0.0)  # One standard deviation below the mean
            print(f"Noisy probes (std. lambda: {std_lambda}) | Acc: {acc:.2f}% | Mean: {loss_mean:.4f} | Std: {loss_std:.4f} | Threshold: {current_loss_thresh:.4f}")
            # current_loss_thresh = np.mean(noisy_stats["loss_vals"]) / 2.  # Half of the mean loss on the noisy probes -- assuming to split the loss into two sets
        else:
            current_loss_thresh = noisy_stats["loss"]  # average loss on the noisy probes
        
        # Compare the numbers between noisy training set and 
        is_noisy = (batch_losses >= current_loss_thresh).to(torch.float32)
        
        model.train()
        num_noisy_ex = int(is_noisy.sum())
        print(f"Noise predictions \t Clean examples: {len(is_noisy)-num_noisy_ex} \t Noisy examples: {num_noisy_ex}")
        return is_noisy

def train_mixUp_HardBootBeta_probes(args, model, device, train_loader, optimizer, epoch, alpha, reg_term, num_classes, probes, std_lambda):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)

        B = compute_is_noisy(data, target, model, probes, std_lambda)
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)


        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


##################### Assign probe class using a combination of GMM and loss values ####################


def assign_probe_class(data, target, model, probes, prob_model, use_gmm=True, adapt_mixture_weights=True, num_modes=3, binary_prediction=True, softmax_probs=False):
    assert num_modes in [2, 3]
    if num_modes == 2:
        assert "noisy" in probes and "typical" in probes, list(probes.keys())
    else:
        assert num_modes == 3
        assert "noisy" in probes and "corrupted" in probes and "typical" in probes, list(probes.keys())
    assert not (binary_prediction and softmax_probs), "Both softmax and binary prediction options cannot be enabled simultaneously..."
    
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses = batch_losses.detach_().cpu().numpy()
        outputs.detach_()
        model.train()
        
        if prob_model is None:
            if use_gmm:
                prob_model = GaussianMixture1D(num_modes=num_modes, learn_mixture_weights=adapt_mixture_weights)
            else:
                prob_model = MultiModalBetaMixture1D(num_modes=num_modes, learn_mixture_weights=adapt_mixture_weights)
            prob_model.fit(model, probes)
            print("Probability model:", prob_model)
        
        # Append the loss values for updating the mixture weights
        prob_model.add_loss_vals(batch_losses)
        
        if num_modes == 3:
            assert binary_prediction, "Only binary prediction is supported when the number of modes is 3"
            predicted_mode = np.array([prob_model.predict(float(x)) for x in batch_losses])
            print(f"GMM predictions \t Clean examples: {np.sum(predicted_mode == 0)} \t Corrupted examples: {np.sum(predicted_mode == 1)} \t Noisy examples: {np.sum(predicted_mode == 2)}")
        else:
            assert num_modes == 2
            if binary_prediction:
                predicted_mode = np.array([prob_model.predict(float(x)) for x in batch_losses])
                print(f"Binary GMM predictions \t Clean examples: {np.sum(predicted_mode == 0)} \t Noisy examples: {np.sum(predicted_mode == 1)}")
            elif softmax_probs:
                predicted_probs = np.array([prob_model.get_softmax_probs(float(x)) for x in batch_losses])
                print(f"Softmax GMM predictions \t Clean examples average prob: {np.sum([x[0] for x in predicted_probs])/len(predicted_probs)} \t Noisy examples average prob: {np.sum([x[1] for x in predicted_probs])/len(predicted_probs)}")
                predicted_mode = np.array([x[1] for x in predicted_probs])  # Probability of a sample being noisy
            else:
                predicted_probs = np.array([prob_model.get_probs(float(x)) for x in batch_losses])
                print(f"GMM predictions \t Clean examples average prob: {np.sum([x[0] for x in predicted_probs])/len(predicted_probs)} \t Noisy examples average prob: {np.sum([x[1] for x in predicted_probs])/len(predicted_probs)}")
                predicted_mode = np.array([x[1] for x in predicted_probs])  # Probability of a sample being noisy
        return torch.from_numpy(predicted_mode), prob_model


def train_mixUp_HardBootBeta_probes_gmm(args, model, device, train_loader, optimizer, epoch, alpha, reg_term, num_classes, probes, prob_model, 
                                        use_gmm, adapt_mixture_weights, binary_prob_model_prediction, softmax_probs, update_model_every_iter):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    recompute_loss_vals = True

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)

        B, prob_model = assign_probe_class(data, target, model, probes, prob_model, use_gmm=use_gmm, adapt_mixture_weights=adapt_mixture_weights, 
                                           num_modes=2, binary_prediction=binary_prob_model_prediction, softmax_probs=softmax_probs)
        if update_model_every_iter:
            prob_model.fit(model, probes)
            print(prob_model)
        
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)


        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    if adapt_mixture_weights and recompute_loss_vals:
        print("Recomputing loss values for updating the mixture weights...")
        prob_model.reset_loss_vals()
        model.eval()
        
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                outputs = model(data)
                outputs = F.log_softmax(outputs, dim=1)
                batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
                batch_losses = batch_losses.detach_().cpu().numpy()
            prob_model.add_loss_vals(batch_losses)
        
        model.train()
    
    # Update the mixture weights based on the collected loss values
    prob_model.update_mixture_weights()
    
    # Update the model itself
    prob_model.fit(model, probes)
    print(prob_model)

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch, prob_model)


##################### Leverage multiple probes simultaneously ####################


def train_mixUp_HardBootBeta_probes_three_sets(args, model, device, train_loader, optimizer, epoch, alpha, reg_term, num_classes, probes, prob_model,
                                               use_gmm, adapt_mixture_weights, update_model_every_iter):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    reweight_loss = False
    adaptive_reweighting = False
    use_flooding = False
    recompute_loss_vals = True

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)

        B, prob_model = assign_probe_class(data, target, model, probes, prob_model, use_gmm=use_gmm, adapt_mixture_weights=adapt_mixture_weights, 
                                           num_modes=3, binary_prediction=True, softmax_probs=False)
        if update_model_every_iter:
            prob_model.fit(model, probes)
            print(prob_model)
        
        # TODO: Compute losses in different ways
        B = B.to(device)
        # B[B <= 1e-4] = 1e-4
        # B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]
        
        # Calculate number of examples from each of these sets for normalization
        frac_clean_ex = (B == 0).sum() / len(B)
        frac_noisy_ex = (B == 2).sum() / len(B)
        frac_corrupted_ex = (B == 1).sum() / len(B)

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        # Original clean (ID == 0)
        # loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1_vec = F.nll_loss(output[B == 0], targets_1[B == 0], reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)
        
        # Original noisy (ID == 2)
        # loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        # loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)
        loss_x1_pred_vec = F.nll_loss(output[B == 2], z1[B == 2], reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(output)

        # Mixup clean (ID == 0)
        # loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        # loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)
        loss_x2_vec = F.nll_loss(output[B2 == 0], targets_2[B2 == 0], reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(output)

        # Mixup noisy (ID == 2)
        # loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        # loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)
        loss_x2_pred_vec = F.nll_loss(output[B2 == 2], z2[B2 == 2], reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(output)
        
        if use_flooding:
            raise NotImplementedError
        
            # Apply flooding loss on third set
            batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
            loss_thresh = torch.zeros_like(batch_losses)
            loss_thresh = batch_losses[B == 1]
        
        # # Reduced loss on the noisy set
        # loss_corrupted_set_vec = F.nll_loss(output[B == 1], targets_1[B == 1], reduction='none')
        # loss_corrupted_set = torch.sum(loss_corrupted_set_vec) / len(loss_corrupted_set_vec)

        # corrupted_lambda = 0.1
        # loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)# + corrupted_lambda * loss_corrupted_set
        
        if reweight_loss:
            if adaptive_reweighting:
                print(f"Fraction of clean ex: {frac_clean_ex} / Fraction of corrupted ex: {frac_corrupted_ex} / Fraction of noisy ex: {frac_noisy_ex}")
                
                loss_x1_reweight = (1. / frac_clean_ex) * loss_x1
                loss_x2_reweight = (1. / frac_clean_ex) * loss_x2
                
                loss_x1_pred_reweight = (1. / frac_noisy_ex) * loss_x1_pred
                loss_x2_pred_reweight = (1. / frac_noisy_ex) * loss_x2_pred
            else:
                loss_x1_reweight = 10. * loss_x1
                loss_x2_reweight = 10. * loss_x2
                
                loss_x1_pred_reweight = 1. * loss_x1_pred
                loss_x2_pred_reweight = 1. * loss_x2_pred
            
            loss = lam*(loss_x1_reweight + loss_x1_pred_reweight) + (1-lam)*(loss_x2_reweight + loss_x2_pred_reweight)
            print(f"Original loss: {loss_x1}/{loss_x2} / Upweighted loss: {loss_x1_reweight}/{loss_x2_reweight}")
        else:
            loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    if adapt_mixture_weights and recompute_loss_vals:
        print("Recomputing loss values for updating the mixture weights...")
        prob_model.reset_loss_vals()
        model.eval()
        
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                outputs = model(data)
                outputs = F.log_softmax(outputs, dim=1)
                batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
                batch_losses = batch_losses.detach_().cpu().numpy()
            prob_model.add_loss_vals(batch_losses)
        
        model.train()
    
    # Update the mixture weights based on the collected loss values
    prob_model.update_mixture_weights()
    
    # Update the model itself
    prob_model.fit(model, probes)
    print(prob_model)

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch, prob_model)


##################### Label correction based on full loss trajectories ####################

def mixup_criterion_per_example(pred, y_a, y_b, lam):
    return lam * F.nll_loss(pred, y_a, reduction='none') + (1 - lam) * F.nll_loss(pred, y_b, reduction='none')


def train_mixUp_traj(args, model, device, train_loader, optimizer, epoch, alpha, trajectory_set):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    
    example_idx = []
    loss_vals = []
    
    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)

        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        # loss_per_ex = mixup_criterion_per_example(output, targets_a, targets_b, lam)
        # loss = loss_per_ex.mean()
        loss = mixup_criterion(output, targets_a, targets_b, lam)
        
        # Compute individual example losses
        with torch.no_grad():
            example_losses = F.nll_loss(output, target, reduction='none')
            example_idx.append(ex_idx.clone().cpu())
            loss_vals.append(example_losses.clone().cpu())

        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
    loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
    
    # Sort the loss list
    sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
    for i in range(len(example_idx)):
        assert sorted_loss_list[example_idx[i]] is None
        sorted_loss_list[example_idx[i]] = loss_vals[i]
    assert not any([x is None for x in sorted_loss_list])
    
    # Append the loss list to loss trajectory
    if trajectory_set is None:
        trajectory_set = dict(train=[sorted_loss_list])
    else:
        assert "train" in trajectory_set
        trajectory_set["train"].append(sorted_loss_list)

    return trajectory_set


def get_set_distances(point_set_A, point_set_B):
    assert len(point_set_A.shape) == 2 and len(point_set_B.shape) == 2, point_set_A.shape
    assert point_set_A.shape[1] == 2 and point_set_B.shape[1] == 2
    return np.linalg.norm(point_set_A[:, None, :] - point_set_B[None, :, :], axis=-1)


def nearest_neighbor_classifier(typical_trajectories, noisy_trajectories, trajectory_set, ex_idx):
    # Preds: 0 -> clean ; 1 -> noisy
    preds = []
    ex_trajs = np.array([trajectory_set[i] for i in ex_idx])
    assert len(ex_trajs.shape) == 2
    assert ex_trajs.shape[1] == typical_trajectories.shape[1] == noisy_trajectories.shape[1]
    typical_probe_dist = np.min(get_set_distances(ex_trajs, typical_trajectories), axis=1)
    noisy_probe_dist = np.min(get_set_distances(ex_trajs, noisy_trajectories), axis=1)
    preds = (noisy_probe_dist < typical_probe_dist).astype(np.int64)  # If the closest example is 
    return torch.from_numpy(preds)


def train_mixUp_HardBootBeta_probes_loss_traj(args, model, device, train_loader, optimizer, epoch, alpha, 
                                              reg_term, num_classes, probes, trajectory_set, use_probs):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    
    example_idx = []
    loss_vals = []
    
    typical_trajectories = np.array(trajectory_set["typical"]).transpose(1, 0)
    noisy_trajectories = np.array(trajectory_set["noisy"]).transpose(1, 0)
    train_trajectories = np.array(trajectory_set["train"]).transpose(1, 0)
    print(f"Typical trajectory size: {typical_trajectories.shape} / Noisy trajectories shape: {noisy_trajectories.shape}")
    print(f"Train trajectories shape: {train_trajectories.shape}")
    
    probe_trajectories = np.concatenate([typical_trajectories, noisy_trajectories], axis=0)
    targets = np.array([0 for _ in range(len(typical_trajectories))] + [1 for _ in range(len(noisy_trajectories))])
    print(f"Combined probe trajectories: {probe_trajectories.shape} / Targets: {targets.shape}")
    
    n_neighbors = 20
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(probe_trajectories, targets)

    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)
        
        # Compute individual example losses
        with torch.no_grad():
            example_losses = F.nll_loss(output, target, reduction='none')
            example_idx.append(ex_idx.clone().cpu())
            loss_vals.append(example_losses.clone().cpu())

        # B = nearest_neighbor_classifier(typical_trajectories, noisy_trajectories, trajectory_set, ex_idx)
        ex_trajs = np.array([train_trajectories[int(i)] for i in ex_idx])
        if use_probs:
            B = clf.predict_proba(ex_trajs)  # 1 means noisy
            assert len(B.shape) == 2 and B.shape[1] == 2, B.shape
            B = B[:, 1]  # Only take the prob for being noisy
        else:
            B = clf.predict(ex_trajs)  # 1 means noisy
        B = torch.from_numpy(np.array(B)).to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)

        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)


        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
    loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
    
    # Sort the loss list
    sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
    for i in range(len(example_idx)):
        assert sorted_loss_list[example_idx[i]] is None
        sorted_loss_list[example_idx[i]] = loss_vals[i]
    assert not any([x is None for x in sorted_loss_list])
    
    # Append the loss list to loss trajectory
    if trajectory_set is None:
        trajectory_set = dict(train=[sorted_loss_list])
    else:
        assert "train" in trajectory_set
        trajectory_set["train"].append(sorted_loss_list)

    typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe")
    noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
    trajectory_set["typical"].append(typical_stats["loss_vals"])
    trajectory_set["noisy"].append(noisy_stats["loss_vals"])

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch, trajectory_set)


def train_mixUp_HardBootBeta_probes_three_sets_loss_traj(args, model, device, train_loader, optimizer, epoch, alpha,
                                                         reg_term, num_classes, probes, trajectory_set, use_probs):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    
    example_idx = []
    loss_vals = []
    
    assert "noisy" in probes and "corrupted" in probes and "typical" in probes, list(probes.keys())
    
    typical_trajectories = np.array(trajectory_set["typical"]).transpose(1, 0)
    noisy_trajectories = np.array(trajectory_set["noisy"]).transpose(1, 0)
    corrupted_trajectories = np.array(trajectory_set["corrupted"]).transpose(1, 0)
    train_trajectories = np.array(trajectory_set["train"]).transpose(1, 0)
    print(f"Typical trajectory size: {typical_trajectories.shape} / Noisy trajectories shape: {noisy_trajectories.shape} / Corrupted trajectories shape: {corrupted_trajectories.shape}")
    print(f"Train trajectories shape: {train_trajectories.shape}")
    
    probe_trajectories = np.concatenate([typical_trajectories, corrupted_trajectories, noisy_trajectories], axis=0)
    targets = np.array([0 for _ in range(len(typical_trajectories))] + [1 for _ in range(len(corrupted_trajectories))] + [2 for _ in range(len(noisy_trajectories))])
    print(f"Combined probe trajectories: {probe_trajectories.shape} / Targets: {targets.shape}")
    
    n_neighbors = 20
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(probe_trajectories, targets)
    use_upweighting = False
    
    use_corrupted_inputs = True
    corrupted_weight = 0.5

    for batch_idx, ((data, target), ex_idx) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)
        
        # Compute individual example losses
        with torch.no_grad():
            example_losses = F.nll_loss(output, target, reduction='none')
            example_idx.append(ex_idx.clone().cpu())
            loss_vals.append(example_losses.clone().cpu())

        # B = nearest_neighbor_classifier(typical_trajectories, noisy_trajectories, trajectory_set, ex_idx)
        ex_trajs = np.array([train_trajectories[int(i)] for i in ex_idx])
        if use_probs:
            B = clf.predict_proba(ex_trajs)  # 1 means noisy
            assert len(B.shape) == 2 and B.shape[1] == 3, B.shape
            
            # Compute the upweighting factor to ensure that the learning rate is not effectively going down with probs
            clean_samples_prob = B[:, 0].sum()
            corrupted_samples_prob = B[:, 1].sum()
            noisy_samples_prob = B[:, 2].sum()
            
            upweight_factor = 1.
            if use_upweighting:  # We are discarding corrupted samples, so the upweighting factor should be based on that
                upweight_factor = (clean_samples_prob + corrupted_samples_prob + noisy_samples_prob) / (clean_samples_prob + noisy_samples_prob)
            
            print(f"Clean prob: {clean_samples_prob/len(B):.4f} / Corrupted prob: {corrupted_samples_prob/len(B):.4f} / Noisy prob: {noisy_samples_prob/len(B):.4f} / Upweighting factor: {upweight_factor:.4f}")
        else:
            B = clf.predict(ex_trajs)  # 1 means noisy
        B = torch.from_numpy(np.array(B)).to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4
        
        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index]
        
        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        # Original clean (ID == 0)
        # loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        if use_probs:
            loss_x1_vec = upweight_factor * B[:, 0] * F.nll_loss(output, targets_1, reduction='none')
        else:
            loss_x1_vec = F.nll_loss(output[B == 0], targets_1[B == 0], reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)
        
        # Original noisy (ID == 2)
        # loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        # loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)
        if use_probs:
            loss_x1_pred_vec = upweight_factor * B[:, 2] * F.nll_loss(output, z1, reduction='none')
        else:
            loss_x1_pred_vec = F.nll_loss(output[B == 2], z1[B == 2], reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(output)

        # Mixup clean (ID == 0)
        # loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        # loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)
        if use_probs:
            loss_x2_vec = upweight_factor * B2[:, 0] * F.nll_loss(output, targets_2, reduction='none')
        else:
            loss_x2_vec = F.nll_loss(output[B2 == 0], targets_2[B2 == 0], reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(output)

        # Mixup noisy (ID == 2)
        # loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        # loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)
        if use_probs:
            loss_x2_pred_vec = upweight_factor * B2[:, 2] * F.nll_loss(output, z2, reduction='none')
        else:
            loss_x2_pred_vec = F.nll_loss(output[B2 == 2], z2[B2 == 2], reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(output)
        
        if use_corrupted_inputs:  # for corrupted inputs
            if use_probs:
                loss_x1_noisy_target_vec = corrupted_weight * B[:, 1] * F.nll_loss(output, targets_1, reduction='none')
                # loss_x1_noisy_pred_vec = corrupted_weight * B[:, 1] * F.nll_loss(output, z1, reduction='none')
                
                loss_x2_noisy_target_vec = corrupted_weight * B2[:, 1] * F.nll_loss(output, targets_2, reduction='none')
                # loss_x2_noisy_pred_vec = corrupted_weight * B2[:, 1] * F.nll_loss(output, z2, reduction='none')
                
            else:
                loss_x1_noisy_target_vec = corrupted_weight * F.nll_loss(output[B == 1], targets_1[B == 1], reduction='none')
                # loss_x1_noisy_pred_vec = corrupted_weight * F.nll_loss(output[B == 1], z1[B == 1], reduction='none')
                
                loss_x2_noisy_target_vec = corrupted_weight * F.nll_loss(output[B2 == 1], targets_2[B2 == 1], reduction='none')
                # loss_x2_noisy_pred_vec = corrupted_weight * F.nll_loss(output[B2 == 1], z2[B2 == 1], reduction='none')

            # loss_x1_noisy_pred = torch.sum(loss_x1_noisy_pred_vec) / len(output)
            # loss_x2_noisy_pred = torch.sum(loss_x2_noisy_pred_vec) / len(output)
            loss_x1_noisy_target = torch.sum(loss_x1_noisy_target_vec) / len(output)
            loss_x2_noisy_target = torch.sum(loss_x2_noisy_target_vec) / len(output)

            loss = lam*(loss_x1 + loss_x1_pred + loss_x1_noisy_target) + (1-lam)*(loss_x2 + loss_x2_pred + loss_x2_noisy_target)
        else:
            loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    example_idx = torch.cat(example_idx, dim=0).numpy().tolist()
    loss_vals = torch.cat(loss_vals, dim=0).numpy().tolist()
    
    # Sort the loss list
    sorted_loss_list = [None for _ in range(len(train_loader.dataset))]
    for i in range(len(example_idx)):
        assert sorted_loss_list[example_idx[i]] is None
        sorted_loss_list[example_idx[i]] = loss_vals[i]
    assert not any([x is None for x in sorted_loss_list])
    
    # Append the loss list to loss trajectory
    if trajectory_set is None:
        trajectory_set = dict(train=[sorted_loss_list])
    else:
        assert "train" in trajectory_set
        trajectory_set["train"].append(sorted_loss_list)

    typical_stats = test_tensor(model, probes["typical"], probes["typical_labels"], msg="Typical probe")
    corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"], msg="Corrupted probe")
    noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"], msg="Noisy probe")
    trajectory_set["typical"].append(typical_stats["loss_vals"])
    trajectory_set["corrupted"].append(corrupted_stats["loss_vals"])
    trajectory_set["noisy"].append(noisy_stats["loss_vals"])

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch, trajectory_set)


##################### Mixup Beta Soft Bootstrapping ####################
# Mixup guided by our beta model with beta soft bootstrapping

def mixup_criterion_mixSoft(pred, y_a, y_b, B, lam, index, output_x1, output_x2):
    return torch.sum(
        (lam) * (
                (1 - B) * F.nll_loss(pred, y_a, reduction='none') + B * (-torch.sum(F.softmax(output_x1, dim=1) * pred, dim=1))) +
                (1-lam) * (
                (1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + B[index] * (-torch.sum(F.softmax(output_x2, dim=1) * pred, dim=1)))) / len(
        pred)


def train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, bmm_model_maxLoss, \
                                            bmm_model_minLoss, reg_term, num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5*torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1-1e-4] = 1-1e-4

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]

        tab_mean_class = torch.mean(output_mean, -2)#Columns mean

        loss = mixup_criterion_mixSoft(output, targets_1, targets_2, B, lam, index, output_x1,
                                                             output_x2)
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term*loss_reg
        loss.backward()


        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################

################################ Dynamic Mixup ##################################
# Mixup guided by our beta model

def mixup_data_beta(x, y, B, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    lam = ((1 - B) + (1 - B[index]))
    mixed_x = ((1-B)/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x + ((1-B[index])/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index

def mixup_criterion_beta(pred, y_a, y_b):
    lam = np.random.beta(32, 32)
    return lam * F.nll_loss(pred, y_a) + (1-lam) * F.nll_loss(pred, y_b)

def train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,
                                bmm_model_maxLoss, bmm_model_minLoss):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5 * torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
        output = model(inputs_mixed)
        output = F.log_softmax(output, dim=1)

        loss = mixup_criterion_beta(output, targets_1, targets_2)

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

################################################################################


################## Dynamic Mixup + soft2hard bootstraping ##################
def mixup_criterion_SoftHard(pred, y_a, y_b, B, index, output_x1, output_x2, Temp):
    return torch.sum(
        (0.5) * (
                (1 - B) * F.nll_loss(pred, y_a, reduction='none') + B * (-torch.sum(F.softmax(output_x1/Temp, dim=1) * pred, dim=1))) +
                (0.5) * (
                (1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + B[index] * (-torch.sum(F.softmax(output_x2/Temp, dim=1) * pred, dim=1)))) / len(
        pred)

def train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, epoch, bmm_model, \
                                    bmm_model_maxLoss, bmm_model_minLoss, countTemp, k, temp_length, reg_term, num_classes):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    steps_every_n = 2 # 2 means that every epoch we change the value of k (index)
    temp_vec = np.linspace(1, 0.001, temp_length)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5*torch.ones(len(target)).float().to(device)
        else:
            B = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1-1e-4] = 1-1e-4

        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]
        tab_mean_class = torch.mean(output_mean,-2)

        Temp = temp_vec[k]

        loss = mixup_criterion_SoftHard(output, targets_1, targets_2, B, index, output_x1, output_x2, Temp)
        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg


        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, Temperature: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr'], Temp))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    countTemp = countTemp + 1
    if countTemp == steps_every_n:
        k = k + 1
        k = min(k, len(temp_vec) - 1)
        countTemp = 1

    return (loss_per_epoch, acc_train_per_epoch, countTemp, k)

########################################################################


def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    cnn_model.eval()
    outputs = cnn_model(data)
    outputs = F.log_softmax(outputs, dim=1)
    batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
    batch_losses.detach_()
    outputs.detach_()
    cnn_model.train()
    batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
    batch_losses[batch_losses >= 1] = 1-10e-4
    batch_losses[batch_losses <= 0] = 10e-4

    #B = bmm_model.posterior(batch_losses,1)
    B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)
    
    print(f"BMM prob predictions \t Clean examples average prob: {np.sum(1-B)/len(B)} \t Noisy examples average prob: {np.sum(B)/len(B)}")
    
    return torch.FloatTensor(B)


def test_cleaning(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    #acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


def compute_loss_set(args, model, device, data_loader):
    model.eval()
    all_losses = torch.Tensor()
    for batch_idx, (data, target) in enumerate(data_loader):
        prediction = model(data.to(device))
        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction.float(), target.to(device), reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
    return all_losses.data.numpy()


def val_cleaning(args, model, device, val_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.val_batch_size))

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.average(acc_val_per_batch)]
    return (loss_per_epoch, acc_val_per_epoch)


################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


class MultiModalBetaMixture1D(object):
    def __init__(self, num_modes: int, learn_mixture_weights: bool=True):
        raise NotImplementedError("BMM models density over unit interval...")
        assert isinstance(num_modes, int)
        self.num_modes = num_modes
        self.weight = np.array([1. / self.num_modes for _ in range(num_modes)])
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        self.learn_mixture_weights = learn_mixture_weights
        
        self.alphas = [None for _ in range(self.num_modes)]
        self.betas = [None for _ in range(self.num_modes)]
        self.key_list = [None for _ in range(self.num_modes)]
        self.loss_list = []
    
    def add_loss_vals(self, loss_vals: np.ndarray):
        self.loss_list.append(loss_vals)
    
    def update_mixture_weights(self):
        if self.learn_mixture_weights:
            losses = np.concatenate(self.loss_list, axis=0)
            assert len(losses.shape) == 1
            print("Losses shape before update:", losses.shape)
            
            # Recompute the mixture weights
            print("Mixture weights before update:", self.weight)
            r = self.responsibilities(losses)
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
            print("Mixture weights after update:", self.weight)
        
        # Reset the loss list
        self.loss_list = []

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(self.num_modes))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(self.num_modes)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, model: torch.nn.Module, probes: dict):
        """
        Fit loss distribution for different probes
        """        
        probe_types = ["typical", "corrupted", "noisy"]
        loss_stats = {}
        for probe in probe_types:
            stats = test_tensor(model, probes[probe], probes[f"{probe}_labels"])
            loss_stats[probe] = stats["loss_vals"]
        probe_class_map = {k: i for i, k in enumerate(probe_types)}
        
        # Fit the BMM distributions based on the loss values
        assert len(probe_types) == self.num_modes
        self.fit_values(loss_stats, probe_class_map)
    
    def fit_values(self, loss_dict: dict, probe_class_map: dict):
        """
        Fit loss distribution for different probes
        :param probe_class_map should map from probe names to their actual label
        """
        self.alphas = [None for _ in range(self.num_modes)]
        self.betas = [None for _ in range(self.num_modes)]
        self.key_list = [None for _ in range(self.num_modes)]
        
        key_list = list(probe_class_map.keys())
        assert len(key_list) == self.num_modes
        
        for k in key_list:
            # Estimate params for beta-distribution
            alpha, beta, _, _ = scipy.stats.beta.fit(loss_dict[k])
            
            class_idx = probe_class_map[k]
            self.key_list[class_idx] = k
            assert 0 <= class_idx < self.num_modes
            self.alphas[class_idx] = alpha
            self.betas[class_idx] = beta
            print(f"Class: {k} / Idx: {class_idx} / Alpha: {alpha:.4f} / Beta: {beta:.4f}")

        assert not any([k is None for k in self.alphas])
        assert not any([k is None for k in self.betas])
        assert not any([k is None for k in self.key_list])
    
    def get_probs(self, x):
        return [self.posterior(x, i) for i in range(self.num_modes)]
    
    def predict(self, x):
        return np.argmax(self.get_probs(x))

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        for i in range(self.num_modes):
            plt.plot(x, self.weighted_likelihood(x, i), label=self.key_list[i])
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'MultiModalBetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


class GaussianMixture1D(object):
    def __init__(self, num_modes: int, learn_mixture_weights: bool=True):
        assert isinstance(num_modes, int)
        assert num_modes in [2, 3], num_modes
        self.num_modes = num_modes
        self.weight = np.array([1. / self.num_modes for _ in range(num_modes)])
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        self.learn_mixture_weights = learn_mixture_weights
        
        self.means = [None for _ in range(self.num_modes)]
        self.stds = [None for _ in range(self.num_modes)]
        self.key_list = [None for _ in range(self.num_modes)]
        self.loss_list = []
    
    def reset_loss_vals(self):
        self.loss_list = []

    def add_loss_vals(self, loss_vals: np.ndarray):
        self.loss_list.append(loss_vals)
    
    def update_mixture_weights(self):
        if self.learn_mixture_weights:
            losses = np.concatenate(self.loss_list, axis=0)
            assert len(losses.shape) == 1
            print("Losses shape before update:", losses.shape)
            
            # Recompute the mixture weights
            print("Mixture weights before update:", self.weight)
            r = self.responsibilities(losses)
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()
            print("Mixture weights after update:", self.weight)
        
        # Reset the loss list
        self.reset_loss_vals()
    
    def likelihood(self, x, y):
        return scipy.stats.norm.pdf(x, self.means[y], self.stds[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)
    
    def weighted_likelihoods(self, x):
        return [self.weighted_likelihood(x, y) for y in range(self.num_modes)]

    def probability(self, x):
        return sum(self.weighted_likelihoods(x))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(self.num_modes)])
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, model: torch.nn.Module, probes: dict):
        """
        Fit loss distribution for different probes
        """        
        probe_types = ["typical", "corrupted", "noisy"] if self.num_modes == 3 else ["typical", "noisy"]
        loss_stats = {}
        for probe in probe_types:
            stats = test_tensor(model, probes[probe], probes[f"{probe}_labels"])
            loss_stats[probe] = stats["loss_vals"]
        probe_class_map = {k: i for i, k in enumerate(probe_types)}
        
        # Fit the GMM distributions based on the loss values
        assert len(probe_types) == self.num_modes
        self.fit_values(loss_stats, probe_class_map)
    
    def fit_values(self, loss_dict: dict, probe_class_map: dict):
        """
        Fit loss distribution for different probes
        :param probe_class_map should map from probe names to their actual label
        """
        self.means = [None for _ in range(self.num_modes)]
        self.stds = [None for _ in range(self.num_modes)]
        self.key_list = [None for _ in range(self.num_modes)]
        
        key_list = list(probe_class_map.keys())
        assert len(key_list) == self.num_modes
        
        for k in key_list:
            loss_vals = loss_dict[k]
            loss_mean = np.mean(loss_vals)
            loss_std = np.std(loss_vals)
            
            class_idx = probe_class_map[k]
            self.key_list[class_idx] = k
            assert 0 <= class_idx < self.num_modes
            self.means[class_idx] = loss_mean
            self.stds[class_idx] = loss_std
            print(f"Class: {k} / Idx: {class_idx} / Loss mean: {loss_mean:.4f} / Loss std: {loss_std:.4f}")
        
        assert not any([k is None for k in self.means])
        assert not any([k is None for k in self.stds])
        assert not any([k is None for k in self.key_list])
    
    def get_probs(self, x):
        return [self.posterior(x, i) for i in range(self.num_modes)]
    
    def predict(self, x):
        return np.argmax(self.get_probs(x))

    def get_softmax_probs(self, x):
        return scipy.special.softmax(self.weighted_likelihoods(x))

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self, output_file):
        if output_file is None:
            return
        
        plt.figure(figsize=(12, 8))
        x = np.linspace(0, 10, 10000)
        for i in range(self.num_modes):
            plt.plot(x, self.weighted_likelihood(x, i), label=self.key_list[i])
        plt.plot(x, self.probability(x), lw=2, label='mixture')
        
        plt.xlabel("Loss value")
        plt.ylabel("Probability density")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)

    def __str__(self):
        return 'GaussianMixture1D(w={}, means={}, stds={}, classes={})'.format(self.weight, self.means, self.stds, self.key_list)
