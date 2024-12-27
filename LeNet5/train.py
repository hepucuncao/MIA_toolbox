import torch  #用来加载数据集
import net
from net import LeNet5
import common
from torchvision import datasets,transforms #导入数据集
import os
import numpy as np 

#数据集中的数据是向量格式，要输入到神经网络中要将数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.ToTensor()
])

train_transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((28, 28)),  # 调整大小为28x28
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Grayscale(),  # 转换为灰度图像
    transforms.Resize((28, 28)), # 调整大小为28x28
    transforms.ToTensor()
])

#加载训练数据集1
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_transform,download=True) #下载手写数字数据集
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次 shuffle：是否打乱

# 加载测试数据集1
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

#如果有显卡，可以转到GPU
device='cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型
model=LeNet5()

#使用优化器
optimizer=common.SGD(model.parameters(),lr=1e-3,momentum=0.9)
#lr:损失率 momentum：动量，主要是应对不同方向上梯度相差大的情况

#学习率每隔10轮，变为原来的0.1，防止训练轮数增多波动太大
lr_scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

#定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n=0.0,0.0,0
    for batch,(X,y) in enumerate(dataloader): #取出数据传入到神经网络中
        #前向传播
        X,y=X.to(device),y.to(device)
        output=model(X)
        cur_loss=common.mean_squared_erro(output,y)  #计算损失：把输出结果和真实值做交叉熵
        _,pred=np.max(output,axis=1) #输出最大概率值
        cur_acc=np.sum(y==pred)/output.shape[0] #求一批次图片精确度的累加和

        optimizer.zero_grad() #反相器梯度清零
        cur_loss.backward()  #反向传播
        optimizer.step()  #梯度更新

        loss+=cur_loss.item()  #把这一批次的loss值累加在一起
        current+=cur_acc.item()
        n=n+1
    print("train_loss:"+str(loss/n))
    print("train_acc:"+str(current/n))

#验证函数
def val(dataloader,model,loss_fn):
    model.eval()
    loss,current,n=0.0,0.0,0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)  # 计算损失：把输出结果和真实值做交叉熵
            _, pred = np.max(output, axis=1)  # 输出最大概率值
            cur_acc = np.sum(y == pred) / output.shape[0]  # 求一批次图片精确度的累加和

            loss += cur_loss.item()  # 把这一批次的loss值累加在一起
            current += cur_acc.item()
            n = n + 1
        print("val_loss:" + str(loss / n))
        print("val_acc:" + str(current / n))
        return current/n

#开始训练
epoch=100  #训练轮次
min_acc=0 #最小精确度
for t in range(epoch):
    print(f'epoch{t+1}\n---------------')
    train(train_dataloader,model,loss_fn,optimizer)
    a=val(test_dataloader,model,loss_fn)
    #保存最好的模型权重9
    if a>min_acc:
        folder='save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc=a
        print('save best model')
        net.save_params(model.state_dict(),'best_model.pth')
print('Done!')
