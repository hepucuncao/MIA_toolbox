import os
import argparse
import utils
import normal
import MIA

def train_networks(args):
    device = utils.get_pytorch_device()
    utils.create_path('./outputs')
    if 'distill' in args.mode:
        model_path_tar = 'networks/{}/{}'.format(0, args.mode.split('_')[-1])
        utils.create_path(model_path_tar)
        model_path_dis = 'networks/{}/{}'.format(args.seed, args.mode)
        utils.create_path(model_path_dis)
    else:
        model_path_tar = 'networks/{}/{}'.format(args.seed, args.mode)
        utils.create_path(model_path_tar)
        model_path_dis = None

    utils.set_logger('outputs/train_models1'.format(args.seed))
    normal.train_models(args, model_path_tar, model_path_dis, device)

def membership_inference_attack(args):
    print(f'--------------{args.mia_type}-------------')

    device = utils.get_pytorch_device()

    if args.mia_type == 'build-dataset':
        models_path = 'networks/{}'.format(0)
        MIA.build_trajectory_membership_dataset(args, models_path, device)

    if args.mia_type == 'black-box':
        trained_models_path = 'networks/{}'.format(args.seed)
        MIA.trajectory_black_box_membership_inference_attack(args, trained_models_path, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryMIA')
    parser.add_argument('--action', type=int, default=0, help=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target', help=['target', 'shadow', 'distill_target', 'distill_shadow'])
    parser.add_argument('--model', type=str, default='resnet', help=['resnet', 'mobilenet', 'vgg', 'wideresnet','lenet','rnn','rl'])
    parser.add_argument('--data', type=str, default='cifar100', help=['cinic10', 'cifar10', 'cifar100', 'gtsrb','mnist'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_distill', type=str, default='resnet', help=['resnet', 'mobilenet', 'vgg', 'wideresnet','lenet','rnn','rl'])
    parser.add_argument('--epochs_distill', type=int, default=100)
    parser.add_argument('--mia_type', type=str, help=['build-dataset', 'black-box'])
    parser.add_argument('--port_num', type=int, default=3)
    
    args = parser.parse_args()
    utils.set_random_seeds(args.seed)
    print('random seed:{}'.format(args.seed))

    if args.action == 0:
        train_networks(args)

    elif args.action == 1:
        membership_inference_attack(args)
    
