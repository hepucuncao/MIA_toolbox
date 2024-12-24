import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import random
from torch.utils.data import DataLoader

# 1.加载必要的库
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 4

# 2.图像处理
pipeline = transforms.Compose([
    transforms.ToTensor(),
])

# 3.下载，加载数据
train_set = datasets.MNIST("../MNIST", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("../MNIST", train=False, download=True, transform=pipeline)

# 4.加载数据集
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# 5.构建网络模型
class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out!= ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        self.blk1 = ResBlk(32, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=1)
        self.blk3 = ResBlk(128, 256, stride=1)
        self.blk4 = ResBlk(256, 256, stride=1)
        self.outlayer = nn.Linear(256 * 1 * 1, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

# 6.定义优化器
model = ResNet18().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 7.训练
def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset), 100. * batch_index / len(train_loader), loss.item()))

# 8.测试
def test_model(model, device, test_loader):
    model.eval()
    correct = 0.0
    text_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            text_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
    text_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss, accuracy))

# 9.删除数据
def delete_data(train_set, ratio):
    indices = list(range(len(train_set)))
    random.shuffle(indices)
    delete_num = int(len(train_set) * ratio)
    delete_indices = indices[:delete_num]
    new_train_set = torch.utils.data.Subset(train_set, [i for i in indices if i not in delete_indices])
    return new_train_set, delete_indices

# 10.主函数
if __name__ == "__main__":
    # 删除5%的数据
    new_train_set_5, delete_indices_5 = delete_data(train_set, 0.05)
    train_loader_5 = DataLoader(new_train_set_5, batch_size=BATCH_SIZE, shuffle=True)
    model_5 = ResNet18().to(DEVICE)
    optimizer_5 = optim.Adam(model_5.parameters(), lr=0.001)
    for epoch in range(1, EPOCHS + 1):
        train_model(model_5, DEVICE, train_loader_5, optimizer_5, epoch)
        test_model(model_5, DEVICE, test_loader)

    # 删除10%的数据
    new_train_set_10, delete_indices_10 = delete_data(train_set, 0.10)
    train_loader_10 = DataLoader(new_train_set_10, batch_size=BATCH_SIZE, shuffle=True)
    model_10 = ResNet18().to(DEVICE)
    optimizer_10 = optim.Adam(model_10.parameters(), lr=0.001)
    for epoch in range(1, EPOCHS + 1):
        train_model(model_10, DEVICE, train_loader_10, optimizer_10, epoch)
        test_model(model_10, DEVICE, test_loader)

    # 保存模型
    torch.save(model.state_dict(), 'MIA_model/model.ckpt')
    torch.save(model_5.state_dict(), 'MIA_model/model_5%.ckpt')
    torch.save(model_10.state_dict(), 'MIA_model/model_10%.ckpt')
