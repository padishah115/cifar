import torch
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import load
import map
import network
from network import training_loop
from network import validation



torch.manual_seed(123)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


def main():
    #Load the training and validation sets
    imgs_train, imgs_val = load.load_sets()
    cifar2, cifar2_val = map.ten_to_two(imgs_train=imgs_train, imgs_val=imgs_val)

    n_epochs = 100
    model = network.softmax_network
    lr = 1e-2
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

    #Perform network training
    training_loop(
        n_epochs=n_epochs,
        model=model,
        optimizer=optimizer,
        loss_fn=nn.NLLLoss(),
        train_loader=train_loader
    )

    #Perform test agsinst validation set
    val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=True)
    accuracy = validation(model=model, val_loader=val_loader)

    print(f'Validation set accuracy: {accuracy}')

if __name__ == '__main__':
    main()

