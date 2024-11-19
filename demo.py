import argparse
from mia import core,utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryMIA')
    parser.add_argument('--action', type=int, default=0, help=[0, 1, 2])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target',
                        help=['target', 'shadow', 'distill_target', 'distill_shadow'])
    parser.add_argument('--model', type=str, default='resnet',
                        help=['resnet', 'mobilenet', 'vgg', 'wideresnet', 'lenet', 'rnn', 'rl'])
    parser.add_argument('--data', type=str, default='cifar100',
                        help=['cinic10', 'cifar10', 'cifar100', 'gtsrb', 'mnist'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_distill', type=str, default='resnet',
                        help=['resnet', 'mobilenet', 'vgg', 'wideresnet', 'lenet', 'rnn', 'rl'])
    parser.add_argument('--epochs_distill', type=int, default=100)
    parser.add_argument('--mia_type', type=str, help=['build-dataset', 'black-box'])
    parser.add_argument('--port_num', type=int, default=3)
    parser.add_argument('--is_detected', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.05)

    args = parser.parse_args()
    if args.action == 0:
        core.train_networks(args)

    elif args.action == 1:
        core.membership_inference_attack(args)


if __name__ == "__main__":
    main()