"""Neural network models for distinguishing between planes and birds, as well as training loops for performing gradient descent"""

import torch
import torch.nn as nn
import torch.optim as optim


def training_loop(n_epochs:int, model:nn.Sequential, optimizer:optim.Optimizer, loss_fn, train_loader:torch.utils.data.DataLoader):
    """Performs training on a batch over some specified number of epochs.
    
    Args:
        n_epochs (int): The number of epochs for which training of the network is performed
        model (nn.Sequential): The neural network to be trained
        optimizer (Optimizer): The optimizer used to perform parameter optimization
        loss_fn: The loss function
        train_loader: DataLoader containing randomly-sampled minibatches from the training set
    
    """

    for epoch in range (1, n_epochs+1):
        for imgs, labels in train_loader:
            batch_size = imgs.shape[0] #number of images in the batch
            outputs = model(imgs.view(batch_size, -1))
            
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad() #Zero the gradients
            loss.backward()
            optimizer.step()

        
        if epoch == 1 or epoch % 50 == 0:
            print(f'Epoch: {epoch}, loss: {loss:.4f}')


def validation(model, val_loader:torch.utils.data.DataLoader)->float:
    """Performs validation over a dataloader containing validation examples.
    
    Args:
        model: The model which has been trained using an optimizer and training set
        val_loader: Validation set for testing

    Returns:
        accuracy (float): Correct diagnoses divded by total number of images tested
    """

    total = 0
    correct = 0

    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        total += batch_size

        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)

        print(f'Predicted labels: {predicted}')
        print(f'True labels: {labels}')

        for predicted, label in zip(predicted, labels):
            if predicted.item() == label.item():
                correct += 1

    return correct / total



#Softmax network- outputs a 2x1 tensor which encodes the estimated bird and plane probabilities
softmax_network = nn.Sequential(
    nn.Linear(3072, 512), #First linear layer
    nn.Tanh(), #First activation function
    nn.Linear(512, 2), #Second linear layer
    nn.LogSoftmax(dim=1) #Softmax output
)