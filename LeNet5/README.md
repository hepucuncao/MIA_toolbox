# LeNet5

2024年5月1日**更新**

在此教程中，我们将对LeNet5模型及其原理进行一个简单的介绍，并实现不依赖库(使用了torch库来加载数据集)的LeNet模型的训练和推理，目前支持MNIST、FashionMNIST和CIFAR-10等数据集，给用户提供一个详细的帮助文档。

## 目录  

[基本介绍](#基本介绍)  
- [LeNet5描述](#LeNet5描述)
- [网络结构](#网络结构)

[LeNet5实现](#LeNet5实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
- [训练及推理步骤](#训练及推理步骤)
- [实例](#实例)


## 基本介绍

### LeNet5描述

说起深度学习目标检算法，就不得不提LeNet-5网络，它是一种经典的卷积神经网络模型。LeNet-5由LeCun等人提出于1998年提出，是一种用于手写体字符识别的非常高效的卷积神经网络。

### 网络结构

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo1.png" width="60%">

LetNet-5是一个较简单的卷积神经网络。上图显示了其结构：输入的二维图像(单通道)，先经过两次卷积层到池化层，再经过全连接层，最后为输出层。整个LeNet-5网络总共包括7层(不含输入层)，分别是：C1、S2、C3、S4、C5、F6、Output。

```
几个参数：
层编号特点：英文字母+数字
英文字母代表以下一种：C→卷积层、S→下采样层(池化)、F→全连接层
数字代表当前是第几层，而非第几卷积层(eg.池化层)

术语解释：
参数→权重w与偏置b
连接数→连线数
参数计算：每个卷积核对应于一个偏置b，卷积核的大小对应于权重w的个数(特别注意通道数)

常用计算公式：N=(W-F+2P)/S+1
其中N：输出大小，W：输入大小，F:卷积核大小，P:填充值的大小(即padding)，S:步长大小(即stride)
```

#### 输入层(Input Layer)

输入层输入的是 32 * 32 像素的图像，注意通道数为1，即单通道灰度图(如果要识别彩色图片，要先把输入转化为单通道灰度图)。

#### C1层

C1层是卷积层，使用6个 5 * 5 大小的卷积核，其中padding=0，stride=1进行卷积，得到6个 28 * 28 大小的特征图，计算公式为：32-5+1=28。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo2.png" width="55%">

**参数个数：** (5 * 5 + 1) * 6 = 156，其中5*5为卷积核的25个参数w，1为偏置项b。

**连接数：** 156 * 28 * 28 = 122304，其中156为单次卷积过程连线数，28*28为输出特征层，每一个像素都由前面卷积得到，即总共经历28*28次卷积。

#### S2层

S2 层是降采样层，使用6个 2 * 2 大小的卷积核进行池化，其中padding=0，stride=2，得到6个 14 * 14 大小的特征图，计算公式为：28/2=14。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo3.png" width="60%">

S2 层其实相当于降采样层+激活层。先是降采样，然后使用激活函数sigmoid非线性输出。先对C1层 2*2 的视野求和，然后进入激活函数，即：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo4.png" width="20%">

**参数个数：** (1 + 1) * 6 = 12，其中第一个1为池化对应的 2 * 2 感受野中最大的那个数的权重w，第二个1为偏置b。

**连接数：** (2 * 2 + 1) * 6 * 14 * 14 = 5880，虽然只选取 2 * 2 感受野之和，但也存在 2 * 2 的连接数，1为偏置项的连接，14 * 14 为输出特征层，每一个像素都由前面卷积得到，即总共经历 14*14次卷积。

#### C3层

C3层是卷积层，使用16个 5 * 5 * n 大小的卷积核，其中padding=0，stride=1进行卷积，得到16个 10 * 10 大小的特征图，计算公式为：14-5+1=10。

16个卷积核并不是都与S2的6个通道层进行卷积操作，如下图所示，C3的前六个特征图(0,1,2,3,4,5)由S2的相邻三个特征图作为输入，对应的卷积核尺寸为：5 * 5 * 3；接下来的6个特征图(6,7,8,9,10,11)由S2的相邻四个特征图作为输入，对应的卷积核尺寸为：5 * 5 * 4；接下来的3个特征图(12,13,14)号特征图由S2间断的四个特征图作为输入，对应的卷积核尺寸为：5 * 5 * 4；最后的15号特征图由S2全部(6个)特征图作为输入，对应的卷积核尺寸为：5 * 5 * 6。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo5.png" width="60%">

```
注意：卷积核大小是 5 *5 且具有3个通道，但每个通道各不相同，这也是下面计算过程中 5 * 5 后面还要乘以3,4,6的原因，这是多通道卷积的计算方法。
```

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo6.png" width="60%">

**参数个数：** (5 * 5 * 3 + 1) * 6 + (5 * 5 * 4 + 1) *6 + (5 * 5 * 4 + 1) * 3 + (5 * 5 * 6 + 1) * 1 = 1516。

**连接数：** 1516 * 10 * 10 = 151600。其中 10*10 为输出特征层，每一个像素都由前面卷积得到，即总共经历 10 * 10 次卷积。

#### S4层

S4层与S2一样也是降采样层，使用16个 2 * 2 大小的卷积核进行池化，其中padding=0，stride=2，得到16个 5 * 5 大小的特征图，计算公式为10/2=5。

**参数个数：** (1 + 1) * 16 = 32。

**连接数：** (2 * 2 + 1) * 16 * 5 * 5 = 2000。

#### C5层

C5层是卷积层，使用120个 5 * 5 * 16 大小的卷积核，padding=0，stride=1进行卷积，得到120个 1 * 1 大小的特征图，计算公式为：5-5+1=1，即相当于 120 个神经元的全连接层。

注意：与C3层不同，这里120个卷积核都与S4的16个通道层进行卷积操作。

**参数个数：** (5 * 5 * 16 + 1) * 120 = 48120。

**连接数：** 48120 * 1 * 1 = 48120。

#### F6层

F6是全连接层，共有84个神经元，与C5层进行全连接，即每个神经元都与C5层的120个特征图相连。计算输入向量和权重向量之间的点积，再加上一个偏置，结果通过sigmoid函数输出。

F6层有84个节点，对应于一个 7 * 12 的比特图，其中-1表示白色，1表示黑色，这样每个符号的比特图的黑白色就对应于一个编码。ASCII 编码图如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo7.png" width="60%">

**参数个数：** (120 + 1) * 84 = 10164。

**连接数：** (120 + 1) * 84 = 10164。

#### Output层

最后的Output层也是全连接层，采用了RBF函数(即径向欧式距离函数)，计算输入向量和参数向量之间的欧式距离(目前已经被Softmax 取代)。

Output层共有10个节点，分别代表数字0到9。假设x是上一层的输入，y是RBF函数的输出，则RBF输出的计算方式如下：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo8.png" width="20%">

上式中i取值范围是0~9，j取值范围是0~7*12-1，w为参数。RBF输出的值越接近0，则越接近i的ASCII 编码图，表示当前网络输入的识别结果是字符i。

下图是数字3的识别过程：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo9.png" width="50%">

**参数个数：** 84 * 10 = 840。

**连接数：** 84 * 10 = 840。


## LeNet5实现

### 总体概述

本项目旨在实现不依赖库的LeNet5模型，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、KMNIST、FashionMNIST数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN、STL-10数据集。模型最终将数据集分类为10种类别，可以根据需要增加分类数量。训练轮次默认为50轮，同样可以根据需要增加训练轮次，对于多通道数据，建议训练轮次在100轮以上，以增大精确度。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/LeNet)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记readme文档，以及其中一些模型的简单实现代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── net.py    # LeNet5模型网络代码
 │  ├── train.py    # LeNet5模型训练代码
 │  └── test.py    # LeNet5模型推理代码
 └── README.md 
```

### 训练及推理步骤

- 1.首先运行net.py初始化LeNet5网络的各个参数
- 2.接着运行train.py进行模型训练，要加载的训练数据集和测试训练集可以自己选择，本项目可以使用的数据集来源于torchvision的datasets库。相关代码如下：

```

train_dataset=datasets.数据集名称(root='保存路径',train=True,transform=data_transform,download=True) #下载手写数字数据集
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

test_dataset = datasets.数据集名称(root='保存路径', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

只需把数据集名称更换成你要使用的数据集(datasets中的数据集)，并修改下载数据集的位置(默认在根目录下，如果路径不存在会自动创建)即可，如果已经提前下载好了则不会下载，否则会自动下载数据集。

```

同时，程序会将每一个训练轮次的训练和验证过程的损失值和精确值打印出来，损失值越接近0、精确值越接近1，则说明训练越成功。

- 3.由于train.py代码会将精确度最高的模型权重保存下来，以便推理的时候直接使用最好的模型，因此运行train.py之前，需要设置好保存的路径，相关代码如下：

```

torch.save(model.state_dict(),'文件名')

默认保存路径为根目录，可以根据需要自己修改路径，该文件夹不存在时会自动创建。

```

- 4.best_model.pth保存完毕后，我们可以运行test.py代码，同样需要加载数据集(和训练过程的数据相同)，步骤同2。同时，我们应将保存的最好模型权重文件加载进来，相关代码如下：

```

model.load_state_dict(torch.load("文件路径"))

文件路径为best_model.pth的路径，注意这里要写绝对路径，并且windows系统要求路径中的斜杠应为反斜杠。

```

另外，程序中创建了一个classes列表来获取分类结果，分类数量由列表中数据的数量来决定，可以根据需要增减，相关代码如下：

```

classes=[
    "0",
    "1",
    ...
    "n-1",
]

要分成n个类别，则写0~n-1个数据项。

```

- 5.最后是推理步骤，程序会选取测试数据集的前n张图片进行推理，并打印出每张图片的预测类别和实际类别，若这两个数据相同则说明推理成功。同时，程序会将选取的图片显示在屏幕上，相关代码如下：

```

for i in range(n): #取前n张图片
    X,y=test_dataset[i][0],test_dataset[i][1]
    show(X).show()
    #把张量扩展为四维
    X=Variable(torch.unsqueeze(X, dim=0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')

推理图片的数量即n取多少可以自己修改，注意要把显示出来的图片手动关掉，程序才会打印出这张图片的预测类别和实际类别。

```

## 实例

这里我们以最经典的MNIST数据集为例：

成功运行完net.py程序后，加载train.py程序的数据集：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo10.png" width="50%">

以及best_model.pth的保存路径：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo11.png" width="50%">

这里我们设置训练轮次为50，由于没有提前下载好数据集，所以程序会自动下载在/data目录下，运行结果如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo12.png" width="50%">

最好的模型权重保存在设置好的路径中：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo13.png" width="30%">

从下图最后一轮的损失值和精确度可以看出，训练的成果已经是非常准确的了！

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo14.png" width="30%">

最后我们运行test.py程序，首先要把train.py运行后保存好的best_model.pth文件加载进来，设置的参数如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo15.png" width="50%">

这里我们设置推理测试数据集中的前20张图片，每推理一张图片，都会弹出来显示在屏幕上，要手动把图片关闭才能打印出预测值和实际值：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo16.png" width="30%">

由下图最终的运行结果我们可以看出，推理的结果是较为准确的，大家可以增加推理图片的数量以测试模型的准确性。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo17.png" width="50%">

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo18.png" width="50%">

其他数据集的训练和推理步骤和MNIST数据集大同小异。


