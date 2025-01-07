import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import argparse

# 2.超参数
BATCH_SIZE = 32#每批处理的数据 一次性多少个
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用GPU
EPOCHS =4 #训练数据集的轮次
# 3.图像处理
pipeline = transforms.Compose([transforms.ToTensor(), #将图片转换为Tensor
])

# 4.下载，加载数据
from torch.utils.data import DataLoader

# 下载数据集
train_set = datasets.MNIST("MNIST",train=True,download=True,transform=pipeline)
test_set = datasets.MNIST("MNIST",train=False,download=True,transform=pipeline)

# 加载数据集 一次性加载BATCH_SIZE个打乱顺序的数据
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)


# 5.构建网络模型
class ResBlk(nn.Module):  # 定义Resnet Block模块
    def __init__(self, ch_in, ch_out, stride=1):  # 进入网络前先得知道传入层数和传出层数的设定
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 初始化

        # we add stride support for resbok, which is distinct from tutorials.
        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # 定义局部向前传播函数
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu
        return out


class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        self.blk1 = ResBlk(32, 64, stride=1)
        self.blk2 = ResBlk(64, 128, stride=1)
        self.blk3 = ResBlk(128, 256, stride=1)
        self.blk4 = ResBlk(256, 256, stride=1)
        self.outlayer = nn.Linear(256 * 1 * 1, 10)  # 最后是全连接层

    def forward(self, x):  # 定义整个向前传播

        x = F.relu(self.conv1(x))  # 先经过第一层卷积

        x = self.blk1(x)  # 然后通过4次resnet网络结构
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # 平铺一维值
        x = self.outlayer(x)  # 全连接层

        return x

# 6.定义优化器
model = ResNet18().to(DEVICE)#创建模型并将模型加载到指定设备上

optimizer = optim.Adam(model.parameters(),lr=0.001)#优化函数

criterion = nn.CrossEntropyLoss()
# 7.训练
def train_model(model,device,train_loader,optimizer,epoch):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()


    model.train()#模型训练
    for batch_index,(data ,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)#部署到DEVICE上去
        optimizer.zero_grad()#梯度初始化为0
        output = model(data)#训练后的结果
        loss = criterion(output,target)#多分类计算损失
        loss.backward()#反向传播 得到参数的梯度值
        optimizer.step()#参数优化
        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))
            if args.dry_run:
                break
# 8.测试
def test_model(model,device,text_loader):
    model.eval()#模型验证
    correct = 0.0#正确率
    global Accuracy
    text_loss = 0.0
    with torch.no_grad():#不会计算梯度，也不会进行反向传播
        for data,target in text_loader:
            data,target = data.to(device),target.to(device)#部署到device上
            output = model(data)#处理后的结果
            text_loss += criterion(output,target).item()#计算测试损失
            pred = output.argmax(dim=1)#找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()#累计正确的值
        text_loss /= len(test_loader.dataset)#损失和/加载的数据集的总数
        Accuracy = 100.0*correct / len(text_loader.dataset)
        print("Test__Average loss: {:4f},Accuracy: {:.3f}\n".format(text_loss,Accuracy))

# 9.调用
for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)

# 10.保存最好参数
torch.save(model.state_dict(),'best_model/model.ckpt')
