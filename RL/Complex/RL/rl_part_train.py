import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from torchvision import datasets, transforms
from net import RL_model
import os


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


# 函数：将数据划分为指定数量的组
def split_data_into_groups(dataset, num_groups):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # 随机打乱数据顺序
    groups = np.array_split(indices, num_groups)  # 划分为 num_groups 个组
    return groups


# 分组重训练函数，只针对被删除的组进行重训练
def incremental_train(model, device, dataset_name, epochs, batch_size, learning_rate, group_indices):
    dataset = MyDataset(dataset_name)
    # 只使用某个被删除的组进行重训练
    subset = Subset(dataset, group_indices)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    print('Incremental training done!')


# 函数：保存模型
def save_model(model, path):
    folder = 'save_model'
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')


# 函数：计算准确率
def evaluate_model(model, device, dataset_name):
    dataset = MyDataset(dataset_name, train=False)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784 if dataset_name == 'mnist' else 3072))
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    return accuracy


# 设置超参数
dataset_name = 'mnist'  # 可以选择'mnist', 'cifar10', 'fashion_mnist'等不同数据类型
epochs = 5  # 针对删除某个组的增量训练，不需要太多epoch
batch_size = 128
learning_rate = 0.001
input_dim = 784 if dataset_name == 'mnist' else 3072
hidden_dim = 256
output_dim = 10

# 初始化模型和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RL_model(input_dim, hidden_dim, output_dim)
model.to(device)

# 加载数据集并将其划分为组
dataset = MyDataset(dataset_name)
groups_5 = split_data_into_groups(dataset, 20)  # 将5%数据分为20组
groups_10 = split_data_into_groups(dataset, 10)  # 将10%数据分为10组

# 保存原始模型
print("Training original model on full dataset")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 完全训练原始模型
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
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

save_model(model, 'save_model/original_model.pth')

# 评估原始模型准确率
original_accuracy = evaluate_model(model, device, dataset_name)
print(f'Original model accuracy: {original_accuracy:.4f}')

# 随机删除5%数据的一个组并进行增量训练
random_group_5 = np.random.choice(len(groups_5))
print(f"Randomly selected group {random_group_5} from 5% groups for removal and retraining.")
incremental_train(model, device, dataset_name, epochs, batch_size, learning_rate, groups_5[random_group_5])
save_model(model, 'save_model/incremental_model_5percent.pth')

# 评估增量训练后的模型准确率
incremental_5_accuracy = evaluate_model(model, device, dataset_name)
print(f'Incremental model (5% data removed) accuracy: {incremental_5_accuracy:.4f}')

# 随机删除10%数据的一个组并进行增量训练
random_group_10 = np.random.choice(len(groups_10))
print(f"Randomly selected group {random_group_10} from 10% groups for removal and retraining.")
incremental_train(model, device, dataset_name, epochs, batch_size, learning_rate, groups_10[random_group_10])
save_model(model, 'save_model/incremental_model_10percent.pth')

# 评估增量训练后的模型准确率
incremental_10_accuracy = evaluate_model(model, device, dataset_name)
print(f'Incremental model (10% data removed) accuracy: {incremental_10_accuracy:.4f}')
