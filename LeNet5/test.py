import torch #用来加载数据集
from net import LeNet5
import numpy as np
from torchvision import datasets,transforms #加载数据集

#数据集中的数据是向量格式，要输入到神经网络中要将数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集1
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_transform,download=True) #下载手写数字数据集
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集1
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_transform,download=True) #下载训练集
test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#如果有显卡，可以转到GPU
device='cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
model=LeNet5().to(device)

#把模型加载进来
model.load_state_dict(load_params("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/LeNet5/save_model/best_model.pth"))
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
def show_image(image):
    Image.fromarray(image.reshape(28, 28).astype('uint8'), 'L').show()


#进入验证
for i in range(20): #取前20张图片
    X,y=test_dataset[i][0],test_dataset[i][1]
    show_image(X)
    #把张量扩展为四维
    X = np.expand_dims(X, axis=0).astype(float) 
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')
