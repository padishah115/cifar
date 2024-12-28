"""Contains functions for loading and normalizing training and validation sets from the CIFAR10 database"""

import torch
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def normalize(imgs:torch.Tensor)->transforms.Normalize:
    RGB_mean = imgs.view(3, -1).mean(dim=1)
    RGB_std = imgs.view(3,-1).std(dim=1)
    return transforms.Normalize(RGB_mean, RGB_std)


def load_sets()->torch.Tensor:
    """Returns the normalised training and validation sets from the CIFAR10 database
    
    Args: None

    Returns:
        imgs_n (Tensor): The normalised training set, made of a stack of images from the CIFAR10 training set
        imgs_val_n (Tensor): The normalised validation set, made of a stack of images from the CIFAR10 validation set
    
    """

    #########################
    #Load the CIFAR database#
    #########################

    #Training Set- unnormalised
    training_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transforms.ToTensor()
        )

    #Creates a tensor stack of images from the training set in order to perform normalisation
    imgs = torch.stack([img for img, _ in training_set], dim=3)


    #########################
    #  Normalise the images #
    #########################

    #Training Set- normalised
    training_set_n = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize(imgs=imgs)
        ])
        )

    #Validation Set- normalised
    validation_set_n = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize(imgs=imgs)
        ])
        )

    return training_set_n, validation_set_n