import numpy as np
import argparse
import os
from classifier_methods import train, iterate_and_shuffle_numpy
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
np.random.seed(171717)

# CONSTANTS
TRAIN_SIZE = 10000
TEST_SIZE = 500

TRAIN_EXAMPLES_AVAILABLE = 50000
TEST_EXAMPLES_AVAILABLE = 10000

MODEL_PATH = './attack_model/'
#DATA_PATH = './CIFAR10'
DATA_PATH = './MNIST'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def generate_data_indices(data_size, target_train_size):
    #Returns indices for data sizing and sampling
    train_indices = np.arange(data_size)
    target_data_indices = np.random.choice(train_indices, target_train_size, replace = False)
    shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    return target_data_indices, shadow_indices

def load_attack_data():
    #Loads presaved training and testing datasets
    fname = MODEL_PATH + 'attack_train_data.pth'
    with np.load(fname) as f:
        train_x, train_y, train_classes = [f['arr_%d' % i]
                                           for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.pth'
    with np.load(fname) as f:
        test_x, test_y, test_classes = [f['arr_%d' % i]
                                        for i in range(len(f.files))]

    return train_x.astype('float32'),train_y.astype('int32'), train_classes.astype('int32'), test_x.astype('float32'), test_y.astype('int32'), test_classes.astype('int32'),


def full_attack_training():

    #divide up dataset between target and shadow models
    #training
    train_indices = list(range(TRAIN_EXAMPLES_AVAILABLE))
    train_target_indices = np.random.choice(train_indices, TRAIN_SIZE, replace=False)
    train_shadow_indices = np.setdiff1d(train_indices, train_target_indices)
    #testing
    test_indices = list(range(TEST_EXAMPLES_AVAILABLE))
    test_target_indices = np.random.choice(test_indices, TEST_SIZE, replace=False)
    test_shadow_indices = np.setdiff1d(test_indices, test_target_indices)

    print("training target model...")
    attack_test_x, attack_test_y, test_classes = train_target_model(
        train_indices=train_target_indices,
        test_indices=test_target_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        model=args.target_model,
        fc_dim_hidden=args.target_fc_dim_hidden,
        save=args.save_model)
    print("done training target model")

    print("training shadow models...")
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        train_indices=train_shadow_indices,
        test_indices=test_shadow_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        fc_dim_hidden=args.target_fc_dim_hidden,
        model=args.target_model,
        save=args.save_model)
    print("done training shadow models")

    print("training attack model...")
    data = (attack_train_x, attack_train_y, train_classes,\
               attack_test_x, attack_test_y, test_classes)
    train_attack_model(
        data=data,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        fc_dim_hidden=args.attack_fc_dim_hidden,
        model=args.attack_model)
    print("done training attack model")

def only_attack_training():
    dataset = None
    train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        fc_dim_hidden=args.attack_fc_dim_hidden,
        model=args.attack_model)

#TRAINING TARGET, SHADOW, ATTACK MODELS
def train_target_model(train_indices, test_indices,
                       epochs = 100, batch_size = 10, learning_rate = 0.01,
                       fc_dim_hidden = 50, model = 'cnn', save = True):

    #CIFAR image preparation
    #transform = transforms.Compose(
    #[transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #trainset=torchvision.datasets.CIFAR10(root='CIFAR10', train = True, download = True,
    #                                        transform = transform)
    trainset = torchvision.datasets.MNIST(root='MNIST', train=True, download=True,
                                            transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, num_workers = 2,
                                              sampler = SubsetRandomSampler(train_indices),
                                              drop_last = True)

    #testset= torchvision.datasets.CIFAR10(root = 'CIFAR10', train = False, download = True,
    #                                       transform = transform)
    testset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True,
                                           transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False,
                                             num_workers = 2,
                                             sampler = SubsetRandomSampler(test_indices),
                                             drop_last = True)


    output_layer, _1, _2 = train(trainloader, testloader,
                                       fc_dim_hidden = fc_dim_hidden, epochs = epochs,
                                       learning_rate = learning_rate, batch_size = batch_size,
                                       model = model)

    # test data for attack model
    attack_x, attack_y, classes = [], [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = output_layer(images)
            attack_x.append(outputs.cpu())
            attack_y.append(np.ones(batch_size))
            classes.append(labels)

        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = output_layer(images)
            attack_x.append(outputs.cpu())
            attack_y.append(np.zeros(batch_size))
            classes.append(labels)

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate([cl.cpu() for cl in classes])

    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = classes.astype('int32')

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_test_data.pth')

    return attack_x, attack_y, classes

def train_shadow_models(train_indices, test_indices,
                        fc_dim_hidden = 50, n_shadow = 20, model = 'cnn',
                        epochs = 100, learning_rate = 0.05, batch_size = 10,
                        save = True):

    #CIFAR image preparation

    #transform = transforms.Compose(
    #[transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #trainset = torchvision.datasets.CIFAR10(root='CIFAR10', train = True, download = True,
    #                                        transform = transform)
    #testset = torchvision.datasets.CIFAR10(root='CIFAR10', train = False, download = True,
    #                                       transform = transform)
    trainset = torchvision.datasets.MNIST(root='MNIST', train=True, download=True,
                                            transform=transform)
    testset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True,
                                           transform=transform)

    attack_x, attack_y, classes = [], [], []
    for i in range(n_shadow):
        print('Training shadow model %d'%(i))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                                   sampler=SubsetRandomSampler(
                                                       np.random.choice(train_indices, TRAIN_SIZE,
                                                                        replace=False)),
                                                   drop_last=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2,
                                                  sampler=SubsetRandomSampler(
                                                      np.random.choice(test_indices,
                                                                       round(TRAIN_SIZE * 0.3),
                                                                       replace=False)),
                                                  drop_last=True)


        output_layer, _1, _2 = train(trainloader, testloader,
                                     fc_dim_hidden=fc_dim_hidden, model=model,
                                     epochs=epochs, learning_rate=learning_rate,
                                     batch_size=batch_size)

        attack_i_x, attack_i_y, classes_i = [], [], []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for data in trainloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                # data used in training, label is 1
                attack_i_y.append(np.ones(batch_size))
                classes_i.append(labels)


            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                # data NOT used in training, label is 0
                attack_i_y.append(np.zeros(batch_size))
                classes_i.append(labels)

        attack_x += attack_i_x
        attack_y += attack_i_y
        classes += classes_i

    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate([cl.cpu() for cl in classes])

    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = classes.astype( 'int32')

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_test_data.pth')

    return attack_x, attack_y, classes

def reduce_ones(x, y, classes):
    ## assumes more training than testing examples
    ## 1 as over-represented class is hardcoded in here
    idx_to_keep = np.where(y == 0)[0]
    idx_to_reduce = np.where(y == 1)[0]
    num_to_reduce = (y.shape[0] - idx_to_reduce.shape[0]) * 2
    idx_sample = np.random.choice(idx_to_reduce, num_to_reduce, replace = False)

    x = x[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    y = y[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    classes = classes[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]

    return x, y, classes

def train_attack_model(data = None,
                       fc_dim_hidden = 50, model = 'fc',
                       learning_rate = 0.01, batch_size = 10, epochs = 3):
    if data is None:
        data = load_attack_data()
    train_x, train_y, train_classes, test_x, test_y, test_classes = data

    ## balance datasets
    train_x, train_y, train_classes = reduce_ones(train_x, train_y, train_classes)
    test_x, test_y, test_classes = reduce_ones(test_x, test_y, test_classes)

    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    true_y = []
    pred_y = []
    for c in unique_classes:
        print('Training attack model for class %d...'%(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        print("training number is %d"%c_train_x.shape[0])
        print("testing number is %d"%c_test_x.shape[0])

        trainloader = iterate_and_shuffle_numpy(c_train_x, c_train_y, batch_size)
        testloader = iterate_and_shuffle_numpy(c_test_x, c_test_y, batch_size)


        _, c_pred_y, c_true_y = train(trainloader, testloader,
                               fc_dim_in = train_x.shape[1],
                               fc_dim_out = 2,
                               fc_dim_hidden = fc_dim_hidden, epochs = epochs, learning_rate = learning_rate,
                               batch_size = batch_size, model = model)
        true_y.append(c_true_y)
        pred_y.append(c_pred_y)
        print("Accuracy score for class %d:"%c)
        print(accuracy_score(c_true_y, c_pred_y))

    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    print('Final full: %0.2f'%(accuracy_score(true_y, pred_y)))
    print(classification_report(true_y, pred_y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## housekeeping
    parser.add_argument('--save_model', type = int, default = 1)
    parser.add_argument('--save_data', type = int, default = 0)

    ## target/blackbox model
    parser.add_argument('--target_model', type = str, default = 'cnn')
    parser.add_argument('--target_learning_rate', type = float, default = 0.001)
    parser.add_argument('--target_batch_size', type = int, default = 4)
    parser.add_argument('--target_fc_dim_hidden', type = int, default = 50)
    parser.add_argument('--target_epochs', type = int, default = 10)

    ## shadow models (intermediate step)
    parser.add_argument('--n_shadow', type = int, default = 10)

    # attack model
    parser.add_argument('--attack_model', type = str, default = 'fc')
    parser.add_argument('--attack_learning_rate', type = float, default = 0.001)
    parser.add_argument('--attack_batch_size', type = int, default = 50)
    parser.add_argument('--attack_fc_dim_hidden', type = int, default = 50)
    parser.add_argument('--attack_epochs', type = int, default = 5)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    if args.save_data:
        save_data()
    else:
        full_attack_training()


