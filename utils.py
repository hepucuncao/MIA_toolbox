import torch
import numpy as np
import random
import sys
import time
import os
import dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from bisect import bisect_right
from normal import save_model

def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True

def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using Pytorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device

class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr
        self.log= open('{}.{}'.format(log_file, mode), "a")
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']

def get_loss_criterion():
    return CrossEntropyLoss()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = dataset.AverageMeter()
    top5 = dataset.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = dataset.accuracy(output, b_y, topk=(1, 5))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc


def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device) 
    b_y = labels.to(device)  
    output = model(b_x)         
    criterion = get_loss_criterion()
    loss = criterion(output, b_y) 
    optimizer.zero_grad()           
    loss.backward()                 
    optimizer.step() 

def cnn_train(args, model, data, epochs, optimizer, scheduler, model_params, model_path, trained_model_name, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        
        cur_lr = get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            if args.mode == 'target':
                print('load aug_target_dataset ... ')
                train_loader = data.aug_target_train_loader
                test_loader = data.aug_target_test_loader
            elif args.mode == 'shadow':
                print('load aug_shadow_dataset ...')
                train_loader = data.aug_shadow_train_loader
                test_loader = data.aug_shadow_test_loader
        else:
            if args.mode == 'target':
                print('load target_dataset ... ')
                train_loader = data.target_train_loader
                test_loader = data.target_test_loader
            elif args.mode == 'shadow':
                print('load shadow_dataset ...')
                train_loader = data.shadow_train_loader
                test_loader = data.shadow_test_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y, idx in train_loader:
            cnn_training_step(model, optimizer, x, y, device)
        end_time = time.time()
    
        top1_test, top5_test = cnn_test(model, test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)
        scheduler.step()
        
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))

    return metrics

def cnn_training_step_dis(model, model_dis, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)   
    b_y_1 = labels.to(device)   
    output = model_dis(b_x)            
    b_y = model(b_x)
    loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(b_y, dim=1))
    optimizer.zero_grad()           
    loss.backward()                 
    optimizer.step() 

def cnn_train_dis(args, model, model_dis, data, epochs, optimizer, scheduler, model_params, model_path, trained_model_name, device='cpu'):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        
        cur_lr = get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            print(f'load aug_{args.mode}_dataset ...')
            train_loader = data.aug_distill_train_loader
            test_loader = data.aug_distill_test_loader 
        else:
            print(f'load {args.mode}_dataset ...')
            train_loader = data.distill_train_loader
            test_loader = data.distill_test_loader

        start_time = time.time()
        model = model.to(device)
        model_dis = model_dis.to(device)
        model_dis.train()
        model.eval()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for i, (x, y, idx)  in enumerate(train_loader):
            cnn_training_step_dis(model, model_dis, optimizer, x, y, device)
        end_time = time.time()
    
        top1_test, top5_test = cnn_test(model_dis, test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)

        top1_train, top5_train = cnn_test(model_dis, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('Top5 Train accuracy: {}'.format(top5_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top5_acc'].append(top5_train)
        epoch_time = int(end_time-start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)
        scheduler.step()
        
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        save_model(model_dis, model_params, model_path, trained_model_name, epoch=epoch)

    return metrics

def get_dataset(dataset, mode, aug=False, batch_size=512, add_trigger=False):
    if dataset == 'cifar10':
        return load_cifar10(mode, aug, batch_size, add_trigger)
    elif dataset == 'gtsrb':
        return load_gtsrb(mode, aug, batch_size, add_trigger)
    elif dataset == 'cinic10':
        return load_cinic10(mode, aug, batch_size, add_trigger)
    elif dataset == 'cifar100':
        return load_cifar100(mode, aug, batch_size)
    elif dataset == 'mnist': 
        return load_mnist(mode, aug, batch_size)
    
def load_gtsrb(mode, aug, batch_size, add_trigger=False):
    gtsrb_data = dataset.GTSRB(mode, aug, batch_size=batch_size)
    return gtsrb_data
    
def load_cinic10(mode, aug, batch_size, add_trigger=False):
    cinic10_data = dataset.CINIC10(mode, aug, batch_size=batch_size, add_trigger=add_trigger)
    return cinic10_data

def load_cifar10(mode, aug, batch_size, add_trigger=False):
    cifar10_data = dataset.CIFAR10(mode, aug, batch_size=batch_size, add_trigger=add_trigger)
    return cifar10_data

def load_cifar100(mode, aug, batch_size):
    cifar100_data = dataset.CIFAR100(mode, aug, batch_size=batch_size)
    return cifar100_data

def load_mnist(mode, aug, batch_size):
    mnist_data = dataset.MNIST(mode, aug, batch_size=batch_size)
    return mnist_data

def get_full_optimizer(model, lr_params, args):
    lr=lr_params[0]
    weight_decay=lr_params[1]
    momentum=lr_params[2]
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    return optimizer, scheduler
