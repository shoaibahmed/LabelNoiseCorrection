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

from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing as preprocessing
import sys
from tqdm import tqdm

import os
import cv2
import kornia.augmentation

import scipy.stats

from style_transfer import style_transfer
from stylized_cifar10.style_transfer_cifar import StyleTransfer

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

def test_tensor(model, data, target, msg=None):
    assert torch.is_tensor(data) and torch.is_tensor(target)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_vals = criterion(output, target)
        test_loss = float(loss_vals.mean())
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = len(data)
    
    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total, 
                       loss_vals=loss_vals.detach().cpu().numpy())
    
    if msg is not None:
        print(f"{msg} | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")
    
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


##################### Leverage multiple probes simultaneously ####################


def assign_probe_class_deprecated(data, target, model, probes, std_lambda=0.0, use_std=False):
    assert std_lambda == 0.0 and not use_std, "Not implemented yet"
    assert "noisy" in probes and "corrupted" in probes, list(probes.keys())
    
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses = batch_losses.detach_().cpu().numpy()
        outputs.detach_()
        
        # Compute noisy probability
        noisy_stats = test_tensor(model, probes["noisy"], probes["noisy_labels"])
        loss_mean = np.mean(noisy_stats["loss_vals"])
        loss_std = np.std(noisy_stats["loss_vals"])
        # prob_mislabeled = scipy.stats.norm(loss_mean, loss_std).pdf(batch_losses)
        mislabeled_dist = scipy.stats.norm(loss_mean, loss_std)
        prob_mislabeled = mislabeled_dist.pdf(batch_losses)
        prob_mislabeled_norm = prob_mislabeled / mislabeled_dist.pdf(loss_mean)
        
        # Compute corrupted probability
        corrupted_stats = test_tensor(model, probes["corrupted"], probes["corrupted_labels"])
        loss_mean = np.mean(corrupted_stats["loss_vals"])
        loss_std = np.std(corrupted_stats["loss_vals"])
        # prob_corrupted = scipy.stats.norm(loss_mean, loss_std).pdf(batch_losses)
        corrupted_dist = scipy.stats.norm(loss_mean, loss_std)
        prob_corrupted = corrupted_dist.pdf(batch_losses)
        prob_corrupted_norm = prob_corrupted / mislabeled_dist.pdf(loss_mean)
        
        prob_clean = (2 - prob_mislabeled_norm - prob_corrupted_norm) / 2
        combined_probs = np.stack([prob_clean, prob_corrupted, prob_mislabeled], axis=1)
        assert combined_probs.shape == (len(prob_clean), 3)
        
        predicted_mode = np.argmax(combined_probs, axis=1)
        assert predicted_mode.shape == (len(prob_clean),)
        
        model.train()
        print(f"Noise predictions \t Clean examples: {np.sum(predicted_mode == 0)} \t Corrupted examples: {np.sum(predicted_mode == 1)} \t Noisy examples: {np.sum(predicted_mode == 2)}")
        return torch.from_numpy(predicted_mode)

def assign_probe_class(data, target, model, probes, gmm=None):
    assert "noisy" in probes and "corrupted" in probes and "typical" in probes, list(probes.keys())
    
    with torch.no_grad():
        model.eval()
        outputs = model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses = batch_losses.detach_().cpu().numpy()
        outputs.detach_()
        
        if gmm is None:
            # Compute noisy probability
            print("Recomputing probe values...")
            probe_types = ["typical", "corrupted", "noisy"]
            loss_stats = {}
            for probe in probe_types:
                stats = test_tensor(model, probes[probe], probes[f"{probe}_labels"])
                loss_stats[probe] = stats["loss_vals"]
            probe_class_map = {k: i for i, k in enumerate(probe_types)}
            
            # Fit the GMM distributions based on the loss values
            gmm = GaussianMixture1D(num_modes=len(probe_class_map))
            gmm.fit_values(loss_stats, probe_class_map)
            print(gmm)
        
        predicted_mode = np.array([gmm.predict(float(x)) for x in batch_losses])
        
        model.train()
        print(f"Noise predictions \t Clean examples: {np.sum(predicted_mode == 0)} \t Corrupted examples: {np.sum(predicted_mode == 1)} \t Noisy examples: {np.sum(predicted_mode == 2)}")
        return torch.from_numpy(predicted_mode), gmm

def train_mixUp_HardBootBeta_probes_three_sets(args, model, device, train_loader, optimizer, epoch, alpha, reg_term, num_classes, probes, std_lambda):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    gmm = None
    update_gmm_every_iter = False
    reweight_loss = False
    adaptive_reweighting = False
    use_flooding = False

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

        B, gmm = assign_probe_class(data, target, model, probes, gmm)
        if update_gmm_every_iter:
            gmm = None
        
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

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)


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

class GaussianMixture1D(object):
    def __init__(self, num_modes):
        assert isinstance(num_modes, int)
        self.num_modes = num_modes
        self.weight = np.array([1. / self.num_modes for _ in range(num_modes)])
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12
        
        self.means = [None for _ in range(self.num_modes)]
        self.stds = [None for _ in range(self.num_modes)]
        self.key_list = [None for _ in range(self.num_modes)]
    
    def likelihood(self, x, y):
        return scipy.stats.norm.pdf(x, self.means[y], self.stds[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(self.num_modes))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, model: torch.nn.Module, probes: dict, probe_class_map: dict):
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
            stats = test_tensor(model, probes[k], probes[f"{k}_labels"], msg=f"{k}_probe", return_loss_vals=True)
            loss_mean = np.mean(stats["loss_vals"])
            loss_std = np.std(stats["loss_vals"])
            # dist = scipy.stats.norm(loss_mean, loss_std)
            
            class_idx = probe_class_map[k]
            self.key_list[class_idx] = k
            assert 0 <= class_idx < self.num_modes
            self.means[class_idx] = loss_mean
            self.stds[class_idx] = loss_std
            print(f"Class: {k} / Idx: {class_idx} / Loss mean: {loss_mean:.4f} / Loss std: {loss_std:.4f}")
        
        assert not any([k is None for k in self.means])
        assert not any([k is None for k in self.stds])
        assert not any([k is None for k in self.key_list])
    
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
        return 'GaussianMixture1D(w={}, means={}, stds={}, classes={})'.format(self.weight, self.means, self.stds, self.key_list)
