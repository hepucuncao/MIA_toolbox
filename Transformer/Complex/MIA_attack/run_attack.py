import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from net import vit_base_patch16_224_in21k as create_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import random

# Set seeds for reproducibility
torch.manual_seed(171717)
np.random.seed(171717)
random.seed(171717)

# CONSTANTS
TRAIN_SIZE = 50000  # 训练集大小
TEST_SIZE = 10000  # 测试集大小
MODEL_PATH = './tf_attack_model/'

num_classes = 100  # CIFAR-100的类别数
batch_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 数据转换
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# 使用torchvision.datasets加载CIFAR-100数据集
train_dataset = datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=data_transform["train"])
val_dataset = datasets.CIFAR100(root='../cifar100', train=False, download=True, transform=data_transform["val"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create necessary directories if they don't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


# Utility function to generate data indices for target and shadow models
def generate_data_indices(data_size, target_size):
    indices = np.arange(data_size)
    target_indices = np.random.choice(indices, target_size, replace=False)
    shadow_indices = np.setdiff1d(indices, target_indices)
    return target_indices, shadow_indices


# Function to randomly delete data from the training set
def remove_data(dataset, percent):
    indices = np.arange(len(dataset))
    num_remove = int(len(dataset) * percent / 100)
    remove_indices = np.random.choice(indices, num_remove, replace=False)
    keep_indices = np.setdiff1d(indices, remove_indices)
    return remove_indices, keep_indices


# Function to train the model
def train_vit_model(model, train_loader, val_loader, epochs=10, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Move data and target to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            out = model(data)

            # Compute loss
            loss = F.nll_loss(out, target)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # Validation phase
    model.eval()
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)

            running_corrects += (pred == target).sum().item()
            total_samples += target.size(0)

    train_acc = running_corrects / total_samples
    test_acc = running_corrects / total_samples  # Assuming val_loader is for test accuracy

    return train_acc, test_acc, model


# Function to prepare attack data
def prepare_attack_data(model, data_loader, train_indices, test_indices):
    model.eval()
    attack_x, attack_y, classes = [], [], []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            logits = model(data)
            for idx in range(len(data)):
                if idx in train_indices:
                    attack_x.append(logits[idx].cpu().numpy())
                    attack_y.append(1)
                    classes.append(target[idx].item())
                elif idx in test_indices:
                    attack_x.append(logits[idx].cpu().numpy())
                    attack_y.append(0)
                    classes.append(target[idx].item())

    attack_x = np.array(attack_x, dtype=np.float32)
    attack_y = np.array(attack_y, dtype=np.int32)
    classes = np.array(classes, dtype=np.int32)

    return attack_x, attack_y, classes


# Function to conduct member inference attacks
def train_attack_model(attack_x, attack_y, epochs=500, batch_size=64):
    scaler = StandardScaler()
    attack_x = scaler.fit_transform(attack_x)

    attack_model = LogisticRegression(max_iter=epochs, solver='lbfgs')
    attack_model.fit(attack_x, attack_y)

    pred_y = attack_model.predict(attack_x)
    acc = accuracy_score(attack_y, pred_y)
    print(f"Attack model accuracy: {acc:.4f}")
    print(classification_report(attack_y, pred_y, zero_division=0))


if __name__ == "__main__":
    # Train target model with full dataset
    target_model = create_model(num_classes=num_classes, has_logits=False).to(device)
    print("Training model with full dataset")
    train_acc, test_acc, target_model = train_vit_model(target_model, train_loader, val_loader)
    attack_x_full, attack_y_full, _ = prepare_attack_data(target_model, train_loader, range(TRAIN_SIZE), range(TEST_SIZE))
    print("Full dataset attack model:")
    train_attack_model(attack_x_full, attack_y_full)

    # Train and attack with 5% data removed
    remove_indices_5, keep_indices_5 = remove_data(train_dataset, 5)
    print(f"Training model with 5% data removed")
    train_loader_5 = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(keep_indices_5))
    train_acc, test_acc, target_model_5 = train_vit_model(target_model, train_loader_5, val_loader)
    attack_x_5, attack_y_5, _ = prepare_attack_data(target_model_5, train_loader_5, keep_indices_5, range(TEST_SIZE))
    print("Attack model for 5% data removed:")
    train_attack_model(attack_x_5, attack_y_5)

    # Train and attack with 10% data removed
    remove_indices_10, keep_indices_10 = remove_data(train_dataset, 10)
    print(f"Training model with 10% data removed")
    train_loader_10 = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(keep_indices_10))
    train_acc, test_acc, target_model_10 = train_vit_model(target_model, train_loader_10, val_loader)
    attack_x_10, attack_y_10, _ = prepare_attack_data(target_model_10, train_loader_10, keep_indices_10, range(TEST_SIZE))
    print("Attack model for 10% data removed:")
    train_attack_model(attack_x_10, attack_y_10)
