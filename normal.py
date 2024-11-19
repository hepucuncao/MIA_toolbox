import os 
import torch
import time 
import random
import numpy as np
import pickle
import utils
from architectures import VGG, MobileNet, ResNet, WideResNet,LeNet,RNN,RL

def train(args, model_path_tar, untrained_model_tar, model_path_dis = None, untrained_model_dis = None, device='cpu'):
    print('Training models...')
    
    if 'distill' in args.mode:
        trained_model, model_params = load_model(args, model_path_dis, untrained_model_dis, epoch=0)
        trained_model_tar, model_params_tar =  load_model(args, model_path_tar, untrained_model_tar, epoch=args.epochs)
    else:
        trained_model, model_params = load_model(args, model_path_tar, untrained_model_tar, epoch=0)
    print(model_params)

    dataset = utils.get_dataset(model_params['task'], args.mode, aug=True)
    learning_rate = model_params['learning_rate']
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    num_epochs = model_params['epochs']
    model_params['optimizer'] = 'SGD'
    optimization_params = (learning_rate, weight_decay, momentum)
    optimizer, scheduler = utils.get_full_optimizer(trained_model, optimization_params, args)

    if 'distill' in args.mode:
        trained_model_name = untrained_model_dis
    else:
        trained_model_name = untrained_model_tar

    print('Training: {}...'.format(trained_model_name))
    trained_model.to(device)
    if 'distill' in args.mode:
        metrics = trained_model.train_func(args, trained_model_tar, trained_model, dataset, num_epochs, optimizer, scheduler, model_params, model_path_dis, trained_model_name, device=device)
    else:
        metrics = trained_model.train_func(args, trained_model, dataset, num_epochs, optimizer, scheduler, model_params, model_path_tar, trained_model_name, device=device)
    model_params['train_top1_acc'] = metrics['train_top1_acc']
    model_params['test_top1_acc'] = metrics['test_top1_acc']
    model_params['train_top5_acc'] = metrics['train_top5_acc']
    model_params['test_top5_acc'] = metrics['test_top5_acc']
    model_params['epoch_times'] = metrics['epoch_times']
    model_params['lrs'] = metrics['lrs']
    total_training_time = sum(model_params['epoch_times'])
    model_params['total_time'] = total_training_time
    print('Training took {} seconds...'.format(total_training_time))
    if 'distill' in args.mode:
        save_model(trained_model, model_params, model_path_dis, trained_model_name, epoch=num_epochs)
    else:
        save_model(trained_model, model_params, model_path_tar, trained_model_name, epoch=num_epochs)

def train_models(args, model_path_tar, model_path_dis, device='cpu'):
    if args.model == 'vgg':
        cnn_tar = create_vgg16bn(model_path_tar, args)
    elif args.model == 'mobilenet':
        cnn_tar = create_mobile(model_path_tar, args)
    elif args.model == 'resnet':
        cnn_tar = create_resnet56(model_path_tar, args)
    elif args.model == 'wideresnet':
        cnn_tar = create_wideresnet32_4(model_path_tar, args)
    elif args.model == 'lenet':
        cnn_tar = create_lenet(model_path_tar, args)
    elif args.model == 'rnn':
        cnn_tar = create_rnn(model_path_tar, args)
    elif args.model == 'rl':
        cnn_tar = create_rl(model_path_tar, args)
        
    if 'distill' in args.mode:
        if args.model == 'vgg':
            cnn_dis = create_vgg16bn(model_path_dis, args)
        elif args.model == 'mobilenet':
            cnn_dis = create_mobile(model_path_dis, args)
        elif args.model == 'resnet':
            cnn_dis = create_resnet56(model_path_dis, args)
        elif args.model == 'wideresnet':
            cnn_dis = create_wideresnet32_4(model_path_dis, args)
        elif args.model == 'lenet':
            cnn_dis = create_lenet(model_path_dis, args)
        elif args.model == 'rnn':
            cnn_dis = create_rnn(model_path_dis, args)
        elif args.model == 'rl':
            cnn_dis = create_rl(model_path_dis, args)
        train(args, model_path_tar, cnn_tar, model_path_dis, cnn_dis, device = device)
    else:
        train(args, model_path_tar, cnn_tar, device=device)

def load_model(args, model_path, model_name, epoch=0):
    model_params = load_params(model_path, model_name, epoch)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture'] 
    network_type = model_params['network_type']

    if 'vgg' in network_type:
        model = VGG(args, model_params)
    elif 'mobilenet' in network_type:
        model = MobileNet(args, model_params)
    elif 'resnet56' in network_type:
        model = ResNet(args, model_params)
    elif 'wideresnet' in network_type:
        model = WideResNet(args,model_params)
    elif 'lenet' in network_type:
        model = LeNet(args,model_params)
    elif 'rnn' in network_type:
        model = RNN(args,model_params)
    elif 'rl' in network_type:
        model = RL(args,model_params)
        
    network_path = model_path + '/' + model_name

    if epoch == 0: 
        load_path = network_path + '/untrained'
    elif epoch == -1: 
        load_path = network_path + '/last'
    else:
        load_path = network_path + '/' + str(epoch)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(load_path), strict=False)
    else:
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)

    return model, model_params

def load_params(models_path, model_name, epoch=0):
    params_path = models_path + '/' + model_name

    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    elif epoch == -1:
        params_path = params_path + '/parameters_last'
    else: 
        params_path = params_path + f'/parameters_{epoch}'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params

def create_vgg16bn(model_path, args):
    print('Creating VGG16BN untrained {} models...'.format(args.data))

    model_params = get_data_params(args.data)
    model_params['fc_layers'] = [512, 512]
    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(args.data)
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True

    get_lr_params(model_params, args)
    model_name = save_networks(args, model_name, model_params, model_path)

    return model_name

def create_mobile(model_path, args):
    print('Creating MobileNet untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_name = '{}_mobilenet'.format(args.data)
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    get_lr_params(model_params, args)
    model_name = save_networks(args, model_name, model_params, model_path)

    return model_name

def create_resnet56(models_path, args):
    print('Creating resnet56 untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]    
    model_name = '{}_resnet56'.format(args.data)
    model_params['network_type'] = 'resnet56'
    model_params['augment_training'] = True 
    model_params['init_weights'] = True
    model_params['port_num'] = args.port_num
    
    get_lr_params(model_params, args)
      
    model_name = save_networks(args, model_name, model_params, models_path)

    return model_name

def create_wideresnet32_4(models_path, args):
    print('Creating wideresnet32_4 untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_params['block_type'] = 'bottle'
    model_params['num_blocks'] = [5,5,5]
    model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3
    model_name = '{}_wideresnet'.format(args.data)
    model_params['network_type'] = 'wideresnet'
    model_params['augment_training'] = True 
    model_params['init_weights'] = True

    get_lr_params(model_params, args)
      
    model_name = save_networks(args, model_name, model_params, models_path)

    return model_name

def create_lenet(models_path, args):
    print('Creating lenet untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_params['block_type'] = 'bottle'
    #model_params['num_blocks'] = [5,5,5]
    #model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3
    model_name = '{}_lenet'.format(args.data)
    model_params['network_type'] = 'lenet'
    model_params['augment_training'] = True 
    model_params['init_weights'] = True

    get_lr_params(model_params, args)
      
    model_name = save_networks(args, model_name, model_params, models_path)

    return model_name

def create_rnn(models_path, args):
    print('Creating rnn untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_params['block_type'] = 'bottle'
    #model_params['num_blocks'] = [5,5,5]
    #model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3
    model_name = '{}_rnn'.format(args.data)
    model_params['network_type'] = 'rnn'
    model_params['augment_training'] = False
    model_params['init_weights'] = True

    get_lr_params(model_params, args)
      
    model_name = save_networks(args, model_name, model_params, models_path)

    return model_name

def create_rl(models_path, args):
    print('Creating rl untrained {} models...'.format(args.data))
    model_params = get_data_params(args.data)
    model_params['block_type'] = 'bottle'
    #model_params['num_blocks'] = [5,5,5]
    #model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3
    model_name = '{}_rl'.format(args.data)
    model_params['network_type'] = 'rl'
    model_params['augment_training'] = True 
    model_params['init_weights'] = True

    get_lr_params(model_params, args)
      
    model_name = save_networks(args, model_name, model_params, models_path)

    return model_name

def save_networks(args, model_name, model_params, model_path):
    print('Saving model...')
    model_params['base_model'] = model_name
    network_type = model_params['network_type']

    if 'vgg' in network_type: 
        model = VGG(args, model_params)
    elif 'mobilenet' in network_type:
        model = MobileNet(args, model_params)
    elif 'resnet56' in network_type:
        model = ResNet(args, model_params)
    elif 'wideresnet' in network_type:
        model = WideResNet(args, model_params)
    elif 'lenet' in network_type:
        model = LeNet(args, model_params)
    elif 'rnn' in network_type:
        model = RNN(args, model_params)
    elif 'rl' in network_type:
        model = RL(args, model_params)
        
    save_model(model, model_params, model_path, model_name, epoch=0)

    return model_name

def save_model(model, model_params, model_path, model_name, epoch=-1):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    network_path = model_path + '/' + model_name
    if not os.path.exists(network_path):
        os.makedirs(network_path)

    if epoch == 0:
        path =  network_path + '/untrained'
        params_path = network_path + '/parameters_untrained'
        torch.save(model.state_dict(), path)

    elif epoch == -1:
        path =  network_path + '/last'
        params_path = network_path + '/parameters_last'
        torch.save(model.state_dict(), path)

    else:
        path = network_path + '/' + str(epoch)
        params_path = network_path + '/parameters_'+str(epoch)
        torch.save(model.state_dict(), path)

    if model_params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)

def get_data_params(data):
    if data == 'cinic10':
        return cinic10_params()
    elif data == 'gtsrb':
        return gtsrb_params()
    elif data == 'cifar10':
        return cifar10_params()
    elif data == 'cifar100':
        return cifar100_params()
    elif data == 'mnist':
        return mnist_params()
    
def gtsrb_params():
    model_params = {}
    model_params['task'] = 'gtsrb'
    model_params['input_size'] = 32
    model_params['num_classes'] = 43
    model_params['port_num'] = 3
    return model_params

def cinic10_params():
    model_params = {}
    model_params['task'] = 'cinic10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    model_params['port_num'] = 3
    return model_params

def cifar10_params():
    model_params = {}
    model_params['task'] = 'cifar10'
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    model_params['port_num'] = 3
    return model_params

def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    model_params['port_num'] = 3
    return model_params

def mnist_params():
    model_params = {}
    model_params['task'] = 'mnist'
    model_params['input_size'] = 28
    model_params['num_classes'] = 10
    model_params['port_num'] = 3
    return model_params

def get_lr_params(model_params, args):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001
    
    model_params['learning_rate'] = 0.1
    model_params['epochs'] = args.epochs
    model_params['scheduler'] = f'CosineAnnealingLR_{args.epochs}'
