'''
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_model import RL_model
from fc_model import FCNet

torch.multiprocessing.set_sharing_strategy('file_system')

def train(trainloader, testloader, model='rl',
          fc_dim_hidden=50, fc_dim_in=10, fc_dim_out=2,
          input_dim=28*28, hidden_dim=128, output_dim=10,
          num_layers=2, dropout_rate=0.2,
          batch_size=10, epochs=10,
          learning_rate=0.001):
    if model == 'fc':
        net = FCNet(dim_hidden=fc_dim_hidden, dim_in=fc_dim_in, dim_out=fc_dim_out,
                    batch_size=batch_size)
    elif model == 'rl':
        net = RL_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       num_layers=num_layers, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Backup data loaders for refreshing
    bak_trainloader, bak_testloader = copy.deepcopy(trainloader), copy.deepcopy(testloader)
    needs_refresh = False
    if hasattr(trainloader, 'needs_refresh'):
        trainloader = bak_trainloader()  # For using iterate_and_shuffle_numpy
        testloader = bak_testloader()
        needs_refresh = True

    for epoch in range(epochs):
        if needs_refresh:
            trainloader = bak_trainloader()
            testloader = bak_testloader()

        running_loss = 0.0
        n_correct = 0
        n_total = 0

        for idx, data in enumerate(trainloader):
            # Get input data and labels
            inputs, labels = data[0], data[1]
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float().to(device)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().to(device)

            # Check and correct the input shape
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Ensure inputs have the correct number of dimensions
            if inputs.dim() == 4:  # Assuming (batch_size, channels, height, width)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            elif inputs.dim() == 2 and inputs.size(1) == input_dim:
                pass  # The input is already in the correct shape
            else:
                raise ValueError(f"Unexpected input shape: {inputs.shape}. Expected shape compatible with (-1, {input_dim}).")

            optimizer.zero_grad()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch == epochs - 1:
            print(f'Epoch: {epoch} Accuracy of the network on the training set: {100 * n_correct / n_total:.2f} %')

        # Evaluate on test set
        n_correct, n_total = 0, 0
        y_hat, y_true = [], []
        with torch.no_grad():
            for idx, data in enumerate(testloader):
                images, labels = data[0], data[1]
                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float().to(device)
                if isinstance(labels, np.ndarray):
                    labels = torch.from_numpy(labels).long().to(device)

                images = images.to(device)
                labels = labels.to(device)

                # Ensure images have the correct number of dimensions
                if images.dim() == 4:  # Assuming (batch_size, channels, height, width)
                    images = images.view(images.size(0), -1)  # Flatten the images
                elif images.dim() == 2 and images.size(1) == input_dim:
                    pass  # The input is already in the correct shape
                else:
                    raise ValueError(f"Unexpected input shape: {images.shape}. Expected shape compatible with (-1, {input_dim}).")

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                n_total += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                if epoch == epochs - 1:
                    y_hat.append(predicted.cpu().numpy())
                    y_true.append(labels.cpu().numpy())

        if epoch == epochs - 1:
            print(f'Epoch: {epoch} Accuracy of the network on the test set: {100 * n_correct / n_total:.2f} %')

    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)
    return net, y_hat, y_true


def iterate_and_shuffle_numpy(inputs, targets, batch_size):
    def return_generator():
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield inputs[excerpt], targets[excerpt]

    return_generator.needs_refresh = True
    return return_generator
'''

from fc_model import FCNet
from rl_model import RL_model

import copy
import numpy as np

import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

def train(trainloader, testloader, model='rl',
          fc_dim_hidden=50, fc_dim_in=10, fc_dim_out=2,
          input_dim=28*28, hidden_dim=128, output_dim=10,
          num_layers=2, dropout_rate=0.2,
          batch_size=10, epochs=10,
          learning_rate=0.001):
    if model == 'fc':
        net = FCNet(dim_hidden=fc_dim_hidden, dim_in=fc_dim_in, dim_out=fc_dim_out,
                    batch_size=batch_size)
    elif model == 'rl':
        net = RL_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                       num_layers=num_layers, dropout_rate=dropout_rate)
    else:
        raise NotImplementedError

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    ## for numpy iteration, need to keep a copy and refresh
    bak_trainloader, bak_testloader = copy.deepcopy(trainloader), copy.deepcopy(testloader)
    needs_refresh = False
    if 'needs_refresh' in dir(trainloader):
        trainloader = bak_trainloader()  ## for case of using iterate_and_shuffle_numpy
        testloader = bak_testloader()
        needs_refresh = True

    for epoch in range(epochs):  # loop over the dataset multiple times

        if needs_refresh:
            trainloader = bak_trainloader()  ## for case of using iterate_and_shuffle_numpy
            testloader = bak_testloader()

        running_loss = 0.0
        n_correct = 0
        n_total = 0

        for idx, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            try:
                inputs, labels = data[0].to(device), data[1].to(device)
            except:
                inputs = torch.from_numpy(data[0]).to(device)
                labels = torch.from_numpy(data[1]).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            try:
                loss = criterion(outputs, labels)
            except:
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

        if epoch == epochs - 1:
            print('Epoch: %d Accuracy of the network on the training set: %d %%' % (
                epoch, 100 * n_correct / n_total))

        n_correct, n_total = 0, 0
        y_hat, y_true = [], []
        with torch.no_grad():
            for idx, data in enumerate(testloader):
                try:
                    images, labels = data[0].to(device), data[1].to(device)
                except:
                    images = torch.from_numpy(data[0]).to(device)
                    labels = torch.from_numpy(data[1]).to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                n_total += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                if epoch == epochs - 1:
                    y_hat.append(predicted.cpu().numpy())
                    y_true.append(labels.cpu().numpy())

        if epoch == epochs - 1:
            print('Epoch: %d Accuracy of the network on the test set: %d %%' % (
                epoch, 100 * n_correct / n_total))

    y_true, y_hat = np.concatenate(y_true), np.concatenate(y_hat)

    return net, y_hat, y_true


def iterate_and_shuffle_numpy(inputs, targets, batch_size):
    def return_generator():
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield inputs[excerpt], targets[excerpt]

    return_generator.needs_refresh = True
    return return_generator

