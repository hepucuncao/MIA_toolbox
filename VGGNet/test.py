import torch
from train import vgg
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage

#数据集中的数据是向量格式，要输入到神经网络中要将数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.Resize([224, 224]),
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

#如果有显卡，可以转到GPU
device='cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
model = vgg(small_conv_arch)

#把模型加载进来
model.load_state_dict(torch.load("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/VGGNet/save_model/best_model.pth1"))
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
    #把张量扩展为四维
    X=Variable(torch.unsqueeze(X, dim=0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')
