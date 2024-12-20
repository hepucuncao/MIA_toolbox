from fc_model import FCNet
from cnn_model import ResNet18

import copy
import numpy as np

import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

def train(trainloader, testloader, model = 'cnn',
          fc_dim_hidden = 50, fc_dim_in = 10, fc_dim_out = 2,
          batch_size = 10, epochs = 10,
          learning_rate = 0.001):

    if model == 'fc':
        net = FCNet(dim_hidden = fc_dim_hidden, dim_in = fc_dim_in, dim_out = fc_dim_out,
                    batch_size = batch_size)
    elif model == 'cnn':
        #net = ConvNet()
        net = ResNet18()
    else:
        raise NotImplementedError
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction = 'mean')
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9)

    ## for numpy iteration, need to keep a copy and refresh 
    bak_trainloader, bak_testloader = copy.deepcopy(trainloader), copy.deepcopy(testloader)
    needs_refresh = False
    if 'needs_refresh' in dir(trainloader):
        trainloader = bak_trainloader() ## for case of using iterate_and_shuffle_numpy
        testloader = bak_testloader()
        needs_refresh = True
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        if needs_refresh:
            trainloader = bak_trainloader() ## for case of using iterate_and_shuffle_numpy
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
        y_hat, y_true  = [], []
        with torch.no_grad():
            for idx, data in enumerate(testloader):
                try:
                    images, labels = data[0].to(device), data[1].to(device)
                except:
                    images = torch.from_numpy(data[0]).to(device)
                    labels = torch.from_numpy(data[1]).to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                n_total   += labels.size(0)
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
