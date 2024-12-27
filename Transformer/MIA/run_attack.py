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

# Set seeds for reproducibility
torch.manual_seed(171717)
np.random.seed(171717)

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


# Train the target model and prepare data for the attack model
def train_target_model(train_loader):
    target_model = create_model(num_classes=num_classes, has_logits=False).to(device)

    train_indices, shadow_indices = generate_data_indices(len(train_loader.dataset), TRAIN_SIZE)
    test_indices, _ = generate_data_indices(len(train_loader.dataset), TEST_SIZE)

    train_acc, test_acc, target_model = train_vit_model(target_model, train_loader, val_loader)
    print(f"Target model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    attack_x, attack_y, classes = prepare_attack_data(target_model, train_loader, train_indices, test_indices)

    np.savez(MODEL_PATH + 'attack_data.npz', attack_x=attack_x, attack_y=attack_y, classes=classes)
    return attack_x, attack_y, classes


# Train the shadow models and prepare data for the attack model
def train_shadow_models(train_loader, n_shadow=10):
    shadow_attack_x, shadow_attack_y, shadow_classes = [], [], []

    for i in range(n_shadow):
        shadow_model = create_model(num_classes=num_classes, has_logits=False).to(device)

        shadow_train_indices, shadow_test_indices = generate_data_indices(len(train_loader.dataset), TRAIN_SIZE)

        # Prepare shadow model data
        shadow_attack_x_temp, shadow_attack_y_temp, shadow_classes_temp = prepare_attack_data(
            shadow_model, train_loader, shadow_train_indices, shadow_test_indices)
        shadow_attack_x.append(shadow_attack_x_temp)
        shadow_attack_y.append(shadow_attack_y_temp)
        shadow_classes.append(shadow_classes_temp)

    shadow_attack_x = np.vstack(shadow_attack_x)
    shadow_attack_y = np.concatenate(shadow_attack_y)
    shadow_classes = np.concatenate(shadow_classes)

    np.savez(MODEL_PATH + 'shadow_attack_data.npz', shadow_attack_x=shadow_attack_x, shadow_attack_y=shadow_attack_y,
             shadow_classes=shadow_classes)
    return shadow_attack_x, shadow_attack_y, shadow_classes


# Train the attack model using the prepared data
def train_attack_model(attack_data=None, epochs=500, batch_size=64, learning_rate=0.001):
    if attack_data is None:
        attack_data = np.load(MODEL_PATH + 'attack_data.npz')
        attack_x, attack_y = attack_data['attack_x'], attack_data['attack_y']
    else:
        attack_x, attack_y = attack_data

    # 标准化数据
    scaler = StandardScaler()
    attack_x = scaler.fit_transform(attack_x)

    # Train attack model (a simple logistic regression)
    attack_model = LogisticRegression(max_iter=epochs, solver='lbfgs')
    attack_model.fit(attack_x, attack_y)

    # Evaluate the attack model
    pred_y = attack_model.predict(attack_x)
    acc = accuracy_score(attack_y, pred_y)
    print(f"Attack model accuracy: {acc:.4f}")

    # Handle undefined precision with zero_division=0
    print(classification_report(attack_y, pred_y, zero_division=0))


if __name__ == "__main__":
    # Train the target model and prepare attack data
    attack_x, attack_y, classes = train_target_model(train_loader)

    # Train shadow models and prepare shadow attack data
    shadow_attack_x, shadow_attack_y, shadow_classes = train_shadow_models(train_loader, n_shadow=10)

    # Train and evaluate the attack model
    train_attack_model((shadow_attack_x, shadow_attack_y))
