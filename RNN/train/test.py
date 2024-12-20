import torch
from net import RNN
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage

#数据集中的数据是向量格式，要输入到神经网络中要将数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集1
train_dataset=datasets.MNIST(root='MNIST',train=True,transform=data_transform,download=True) #下载手写数字数据集
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集1
test_dataset=datasets.MNIST(root='MNIST',train=False,transform=data_transform,download=True) #下载训练集
test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#加载训练数据集2
#train_dataset=datasets.FashionMNIST(root='./data1',train=True,transform=data_transform,download=True) #下载手写数字数据集
#train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集2
#test_dataset=datasets.FashionMNIST(root='./data1',train=False,transform=data_transform,download=True) #下载训练集
#test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#加载训练数据集3
#train_dataset=datasets.KMNIST(root='./data3',train=True,transform=data_transform,download=True) #下载手写数字数据集
#train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集3
#test_dataset=datasets.KMNIST(root='./data3',train=False,transform=data_transform,download=True) #下载训练集
#test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#加载训练数据集4
#train_dataset=datasets.CIFAR10(root='./data2',train=True,transform=data_transform,download=True) #下载手写数字数据集
#train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集4
#test_dataset=datasets.CIFAR10(root='./data2',train=False,transform=data_transform,download=True) #下载训练集
#test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#如果有显卡，可以转到GPU
device='cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
input_dim = 28 #MNIST数据集里的图片是28×28的图，所以输入为28，相当于每次的输入是图片的一行
hidden_dim = 128
layer_dim = 1 #模型的层数（除开最后的全连接层）
output_dim = 10 #最后识别的是0-9这10种数字，相当于分为10类
MyRNN = RNN(input_dim,hidden_dim,layer_dim,output_dim) #将上述参数传入RNNimc完成实例化
model = MyRNN.to(device)

#把模型加载进来
model.load_state_dict(torch.load("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/RNN/save_model/best_model.pth"))
#写绝对路径 win系统要求改为反斜杠

#获取结果
classes=[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

#把tensor转化为图片，方便可视化
show=ToPILImage()

#进入验证
for i in range(20): #取前20张图片
    X,y=test_dataset[i][0],test_dataset[i][1]
    show(X).show()
    # 把张量扩展为三维
    X = Variable(X.float().view(1, 28, -1), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')



