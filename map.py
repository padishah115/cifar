"""Module responsible for extracting the bird and airplane images from the CIFAR10 dataset"""

import torch

def ten_to_two(imgs_train:torch.Tensor, imgs_val:torch.Tensor)->torch.Tensor:
    """Downsizes the CIFAR10 database to only include two classes- for birds and airplanes
    
    Args:
        imgs_train (Tensor): The CIFAR10 training set
        imgs_val (Tensor): The CIFAR10 validation set

    Returns:
        cifar2 (Tensor): The two-class training set
        cifar2_val (Tensor): The two-class validation set

    """

    label_map = {0:0, 2:1}

    cifar2 = [(img, label_map[label]) for img, label in imgs_train if label in [0, 2]]
    cifar2_val = [(img, label_map[label]) for img, label in imgs_val if label in [0, 2]]

    return cifar2, cifar2_val