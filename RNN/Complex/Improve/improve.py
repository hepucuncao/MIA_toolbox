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
DATA_PATH = './MNIST'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

def generate_data_indices(data_size, target_train_size):
    train_indices = np.arange(data_size)
    target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
    shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    return target_data_indices, shadow_indices

def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.pth'
    with np.load(fname) as f:
        train_x, train_y, train_classes = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.pth'
    with np.load(fname) as f:
        test_x, test_y, test_classes = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), train_classes.astype('int32'), test_x.astype('float32'), test_y.astype('int32'), test_classes.astype('int32')

def full_attack_training():
    train_indices = list(range(TRAIN_EXAMPLES_AVAILABLE))
    train_target_indices = np.random.choice(train_indices, TRAIN_SIZE, replace=False)
    train_shadow_indices = np.setdiff1d(train_indices, train_target_indices)

    test_indices = list(range(TEST_EXAMPLES_AVAILABLE))
    test_target_indices = np.random.choice(test_indices, TEST_SIZE, replace=False)
    test_shadow_indices = np.setdiff1d(test_indices, test_target_indices)

    print("Training target model with improved settings...")
    attack_test_x, attack_test_y, test_classes = train_target_model(
        train_indices=train_target_indices,
        test_indices=test_target_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        model=args.target_model,
        fc_dim_hidden=args.target_fc_dim_hidden,
        save=args.save_model
    )
    print("Target model training complete.")

    print("Training shadow models with improved settings...")
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        train_indices=train_shadow_indices,
        test_indices=test_shadow_indices,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        fc_dim_hidden=args.target_fc_dim_hidden,
        model=args.target_model,
        save=args.save_model
    )
    print("Shadow model training complete.")

    print("Training attack model with improved settings...")
    data = (attack_train_x, attack_train_y, train_classes, attack_test_x, attack_test_y, test_classes)
    train_attack_model(
        data=data,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        fc_dim_hidden=args.attack_fc_dim_hidden,
        model=args.attack_model
    )
    print("Attack model training complete.")

def train_target_model(train_indices, test_indices, epochs=100, batch_size=10, learning_rate=0.01, fc_dim_hidden=100, model='rnn', save=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='MNIST', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, sampler=SubsetRandomSampler(train_indices), drop_last=True)

    testset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=SubsetRandomSampler(test_indices), drop_last=True)

    output_layer = train(trainloader, testloader, fc_dim_hidden=fc_dim_hidden, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, model=model)

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

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_test_data.pth')

    return attack_x, attack_y, classes

def train_shadow_models(train_indices, test_indices, fc_dim_hidden=100, n_shadow=100, model='rnn', epochs=100, learning_rate=0.001, batch_size=10, save=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(root='MNIST', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='MNIST', train=False, download=True, transform=transform)

    attack_x, attack_y, classes = [], [], []
    for i in range(n_shadow):
        print(f'Training shadow model {i} with RNN...')
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, sampler=SubsetRandomSampler(np.random.choice(train_indices, TRAIN_SIZE, replace=False)), drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=SubsetRandomSampler(np.random.choice(test_indices, round(TRAIN_SIZE * 0.3), replace=False)), drop_last=True)

        output_layer = train(trainloader, testloader, fc_dim_hidden=fc_dim_hidden, model=model, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        attack_i_x, attack_i_y, classes_i = [], [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for data in trainloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                attack_i_y.append(np.ones(batch_size))
                classes_i.append(labels)

            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = output_layer(images)
                attack_i_x.append(outputs.cpu())
                attack_i_y.append(np.zeros(batch_size))
                classes_i.append(labels)

        attack_x += attack_i_x
        attack_y += attack_i_y
        classes += classes_i

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate([cl.cpu() for cl in classes])

    if save:
        torch.save((attack_x, attack_y, classes), MODEL_PATH + 'attack_train_data.pth')

    return attack_x.astype('float32'), attack_y.astype('int32'), classes.astype('int32')

def train_attack_model(data, fc_dim_hidden=100, epochs=100, learning_rate=0.001, batch_size=10, model='rnn'):
    attack_train_x, attack_train_y, attack_train_classes, attack_test_x, attack_test_y, attack_test_classes = data

    attack_train_x, attack_train_y = iterate_and_shuffle_numpy(attack_train_x, attack_train_y, batch_size)
    attack_test_x, attack_test_y = iterate_and_shuffle_numpy(attack_test_x, attack_test_y, batch_size)

    trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(attack_train_x),
                                                                            torch.from_numpy(attack_train_y).view(-1)),
                                              batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(attack_test_x),
                                                                           torch.from_numpy(attack_test_y).view(-1)),
                                             batch_size=batch_size, shuffle=False)

    output_layer = train(trainloader, testloader, fc_dim_hidden=fc_dim_hidden, model=model, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

    attack_preds, attack_true = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = output_layer(inputs)
            predicted = (outputs > 0.5).long().cpu()
            attack_preds.append(predicted)
            attack_true.append(labels)

    attack_preds = np.concatenate(attack_preds)
    attack_true = np.concatenate(attack_true)

    print("Attack model accuracy: ", accuracy_score(attack_true, attack_preds))
    print("Classification report: \n", classification_report(attack_true, attack_preds))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_batch_size', type=int, default=50)
    parser.add_argument('--target_learning_rate', type=float, default=0.0001)
    parser.add_argument('--target_fc_dim_hidden', type=int, default=128)
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_learning_rate', type=float, default=0.0001)
    parser.add_argument('--attack_fc_dim_hidden', type=int, default=64)
    parser.add_argument('--target_model', type=str, default='rnn', choices=['fc', 'rnn'])
    parser.add_argument('--attack_model', type=str, default='rnn', choices=['fc', 'rnn'])
    parser.add_argument('--save_model', action='store_true', default=True)
    global args
    args = parser.parse_args()

    full_attack_training()

if __name__ == "__main__":
    main()
