import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from net import RL_model
import os
import hashlib
import time


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, dataset_name, train=True):
        if dataset_name == 'mnist':
            self.dataset = datasets.MNIST('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                          download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                            download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == 'fashion_mnist':
            self.dataset = datasets.FashionMNIST('~/PycharmProjects/pythonProject1/DeepLearning/RL',
                                                 download=True, train=train, transform=transforms.ToTensor())
        else:
            raise ValueError('Unsupported dataset')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label


# 函数：随机删除数据并记录被删除的数据索引
def remove_random_data(dataset, remove_percent):
    num_samples = len(dataset)
    remove_count = int(remove_percent * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    remove_indices = indices[:remove_count]
    remaining_indices = indices[remove_count:]
    # 返回删除的数据索引和剩余的数据
    return remove_indices, remaining_indices


# 函数：计算模型参数的哈希值
def compute_model_hash(model):
    # 获取模型参数
    model_params = torch.cat([p.view(-1) for p in model.parameters()])
    # 初始化随机数
    random_tensor = torch.randn_like(model_params)
    # 计算点积
    dot_product = torch.dot(model_params, random_tensor)

    # 开始计时
    start_time = time.time()

    # 计算 SHA-256 哈希值
    hash_value = hashlib.sha256(dot_product.detach().cpu().numpy()).hexdigest()

    # 计算耗时
    elapsed_time = time.time() - start_time
    return hash_value, elapsed_time


# 训练模型函数
def train(model, device, dataset_name, epochs, batch_size, learning_rate, val_size=0.1, train_indices=None):
    dataset = MyDataset(dataset_name)

    if train_indices is not None:
        dataset = Subset(dataset, train_indices)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 打印每个 epoch 的 loss
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        print(f'Validation Accuracy: {accuracy:.4f}')

        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            folder = 'save_model'
            if not os.path.exists(folder):
                os.mkdir('save_model')

            # 根据是否提供了train_indices生成模型文件名
            if train_indices is not None:
                model_path = f'save_model/best_model_{len(train_indices)}.pth'
            else:
                model_path = 'save_model/best_model_full.pth'

            print(f'saving best model to {model_path}')
            torch.save(model.state_dict(), model_path)

    print('Done!')


# 推理模型函数
def inference(model, device, dataset_name):
    dataset = MyDataset(dataset_name, train=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
    accuracy = correct / len(dataset)
    print(f'Test Accuracy: {accuracy:.4f}')


# 设置超参数
dataset_name = 'mnist'  # 可以选择'mnist', 'cifar10', 'fashion_mnist'等不同数据类型
epochs = 20
batch_size = 128
learning_rate = 0.001
input_dim = 784 if dataset_name == 'mnist' else 3072
hidden_dim = 256
output_dim = 10

# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RL_model(input_dim, hidden_dim, output_dim)
model.to(device)

# 加载数据集并记录删除的数据
dataset = MyDataset(dataset_name)
remove_5_indices, remaining_5_indices = remove_random_data(dataset, 0.05)
remove_10_indices, remaining_10_indices = remove_random_data(dataset, 0.10)

# 保存删除的数据索引
np.save('removed_5_percent.npy', remove_5_indices)
np.save('removed_10_percent.npy', remove_10_indices)

# 训练原始模型
print("Training on full dataset")
train(model, device, dataset_name, epochs, batch_size, learning_rate)

# 计算并打印模型参数的哈希值
hash_value, elapsed_time = compute_model_hash(model)
print(f'Model Hash: {hash_value}, Time taken: {elapsed_time:.6f} seconds')

# 训练删除5%数据后的模型
print("Training on dataset with 5% data removed")
model_5 = RL_model(input_dim, hidden_dim, output_dim).to(device)
train(model_5, device, dataset_name, epochs, batch_size, learning_rate, train_indices=remaining_5_indices)

# 计算并打印模型参数的哈希值
hash_value_5, elapsed_time_5 = compute_model_hash(model_5)
print(f'Model Hash (5% removed): {hash_value_5}, Time taken: {elapsed_time_5:.6f} seconds')

# 训练删除10%数据后的模型
print("Training on dataset with 10% data removed")
model_10 = RL_model(input_dim, hidden_dim, output_dim).to(device)
train(model_10, device, dataset_name, epochs, batch_size, learning_rate, train_indices=remaining_10_indices)

# 计算并打印模型参数的哈希值
hash_value_10, elapsed_time_10 = compute_model_hash(model_10)
print(f'Model Hash (10% removed): {hash_value_10}, Time taken: {elapsed_time_10:.6f} seconds')

# 推理模型
inference(model, device, dataset_name)
inference(model_5, device, dataset_name)
inference(model_10, device, dataset_name)

