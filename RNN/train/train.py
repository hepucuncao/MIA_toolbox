import torch
from torch import nn
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from net import RNN
import os

# 定义对数据的变换操作
transform = transforms.Compose([
    # 先对图像的尺寸进行修改，然后再转换成张量
    transforms.Resize([28,28]),
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((28,28)),  # 调整大小为28*28
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((28,28)), # 调整大小为28*28
    transforms.ToTensor()
])

#MNIST数据集
train_data = torchvision.datasets.MNIST(root='MNIST',train=True,transform=transform,download=True)
train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)

test_data = torchvision.datasets.MNIST(root='MNIST',train=False,transform=transform,download=True)
test_loader = Data.DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0)

#CIFAR-10数据集
#train_data = torchvision.datasets.CIFAR10(root='CIFAR10',train=True,transform=train_transform,download=True)
#train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)

#test_data = torchvision.datasets.CIFAR10(root='CIFAR10',train=False,transform=test_transform,download=True)
#test_loader = Data.DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0)

input_dim = 28 #MNIST数据集里的图片是28×28的图，所以输入为28，相当于每次的输入是图片的一行
hidden_dim = 128
layer_dim = 1 #模型的层数（除开最后的全连接层）
output_dim = 10 #最后识别的是0-9这10种数字，相当于分为10类
MyRNN = RNN(input_dim,hidden_dim,layer_dim,output_dim) #将上述参数传入RNNimc完成实例化

optimizer = torch.optim.RMSprop(MyRNN.parameters(),lr=0.0003) #优化器的选择
criterion = nn.CrossEntropyLoss() #使用交叉熵损失更新参数
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []
num_epochs = 20

for epoch in range(num_epochs):
    print("Epoch {}:".format(epoch,num_epochs))
    MyRNN.train() ##设置模型为训练模式
    corrects = 0
    train_num = 0
    for step,(b_x,b_y) in enumerate(train_loader):
        xdata = b_x.view(-1,28,28) #将我们的输入reshape，方便传入网络
        output = MyRNN(xdata) #输出为x通过网络的结果
        pre_lab = torch.argmax(output,dim=1) #找出这个一维向量(按行)里面最大值的索引。可以理解为，输出是一个1×10的向量，每个数值都是其是0-9数字的概率，找到概率最大的就是其预测的值
        loss = criterion(output,b_y) #计算输出与真实值的损失函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss += loss.item()*b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        train_num += b_x.size(0)
    #计算经过一个epoch的训练后在训练集上的损失和精度
    train_loss_all.append(loss/train_num)
    train_acc_all.append(corrects.double().item()/train_num)
    print("{} Train Loss:{:.4f} Train Acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))

    min_acc = 0
    #设置为验证模式
    MyRNN.eval()
    corrects = 0
    test_num = 0
    for step,(b_x,b_y) in enumerate(test_loader):
        xdata = b_x.view(-1,28,28)
        output = MyRNN(xdata)
        pre_lab = torch.argmax(output,1)
        loss = criterion(output,b_y) #计算输出与真实值的损失函数
        loss += loss.item()*b_x.size(0)
        corrects += torch.sum(pre_lab == b_y.data)
        test_num += b_x.size(0)
    #计算经过一个epoch的训练后在训练集上的损失和精度
    test_loss_all.append(loss / test_num)
    test_acc_all.append(corrects.double().item() / test_num)
    print("{} Test Loss:{:.4f} Test Acc:{:.4f}".format(epoch, test_loss_all[-1], test_acc_all[-1]))

    # 保存最好的模型权重
    if loss > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = loss
        print('save best model')
        torch.save(MyRNN.state_dict(), 'save_model/best_model.pth')
