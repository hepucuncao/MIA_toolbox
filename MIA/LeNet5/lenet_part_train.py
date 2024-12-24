import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os
import random

# 数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./MNIST', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./MNIST', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

# 定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向转播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("train_loss:" + str(loss/n))
    print("train_acc:" + str(current/n))

def val(dataloader, model, loss_fn):
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 前向转播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss:" + str(loss / n))
        print("val_acc:" + str(current / n))

        return current/n

# 开始训练
epoch = 20
min_acc = 0

for t in range(epoch):
    print(f'epoch{t+1}\n------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'MIA_model'
        if not os.path.exists(folder):
            os.mkdir('MIA_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'MIA_model/best_model1.pth')
print('Done!')

# 删除特定组的数据
# 假设我们将数据集分为20个组，每个组包含5%的数据
# 假设我们将数据集分为10个组，每个组包含10%的数据
#group_size = len(train_dataset) // 10
group_size = len(train_dataset) // 20
removed_group = random.randint(0, 9)
remaining_indices = [i for idx, i in enumerate(range(len(train_dataset))) if idx // group_size != removed_group]
remaining_train_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
train_dataloader_partial = torch.utils.data.DataLoader(remaining_train_dataset, batch_size=16, shuffle=True)

# 训练模型，使用剩余的数据
for t in range(epoch):
    print(f'epoch{t+1} with group {removed_group} removed\n------------------')
    train(train_dataloader_partial, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'MIA_model'
        if not os.path.exists(folder):
            os.mkdir('MIA_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'MIA_model/best_model_partial.pth')
