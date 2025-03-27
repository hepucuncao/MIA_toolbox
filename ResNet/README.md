# ResNet

2024年5月7日**更新**

在此教程中，我们将对ResNet模型及其原理进行一个简单的介绍，并实现ResNet模型的训练和推理，目前支持数据集有：MNIST、fashionMNIST、CIFAR10等，并给用户提供一个详细的帮助文档。

## 目录  

[基本介绍](#基本介绍)  
- [ResNet描述](#ResNet描述)
- [为什么要引入ResNet?](#为什么要引入ResNet?)
- [网络结构分析](#网络结构分析)

[ResNet实现](#ResNet实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
- [训练及推理步骤](#训练及推理步骤)
- [实例](#实例)


## 基本介绍

### ResNet描述

ResNet是一种残差网络，咱们可以把它理解为一个子网络，这个子网络经过堆叠可以构成一个很深的网络。下面是一个简单的ResNet结构：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo1.png" width="50%">

### 为什么要引入ResNet？

通过前面的学习我们知道，网络越深，咱们能获取的信息越多，而且特征也越丰富。但是根据实验表明，随着网络的加深，优化效果反而越差，测试数据和训练数据的准确率反而降低了，这是由于网络的加深会造成梯度爆炸和梯度消失的问题。如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo2.png" width="60%">

为了让更深的网络也能训练出好的效果，一个新的网络结构——ResNet出现了。

### 网络结构分析

ResNet block有两种，一种两层结构，一种三层结构，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo3.png" width="60%">

现在要求解的映射为：H(x)，将这个问题转换为求解网络的残差映射函数，也就是F(x)，其中F(x) = H(x)-x。
```

残差：观测值与估计值之间的差。
这里H(x)就是观测值，x就是估计值(也就是上一层ResNet输出的特征映射)。
一般称x为identity Function，它是一个跳跃连接；称F(x)为ResNet Function。

```

于是要求解的问题变成了H(x) = F(x)+x。

关于为什么要经过F(x)之后再求解H(x)，相信很多人会有疑问。如果是采用一般的卷积神经网络的化，原先要求解的是H(x) = F(x)这个值，那么现在假设，在网络中达到某一个深度时已经达到最优状态了，也就是说，此时的错误率是最低的时候，再往下加深网络的化就会出现退化问题(即错误率上升)。现在要更新下一层网络的权值就会变得很麻烦，因为权值得是一个让下一层网络同样也是最优状态才行。

但是采用残差网络就能很好的解决这个问题。仍然假设当前网络的深度能够使得错误率最低，如果继续增加ResNet，为了保证下一层的网络状态仍然是最优状态，只需要令F(x)=0就可以。因为x是当前输出的最优解，为了让它成为下一层的最优解也就是输出H(x)=x的话，只要让F(x)=0就行了。

当然上面提到的只是理想情况，在真实测试的时候x肯定是很难达到最优的，但是总会有那么一个时刻它能够无限接近最优解。这时采用ResNet的话，也只用小小的更新F(x)部分的权重值就行了，而不用像一般的卷积层一样大动干戈。

```

注意：如果残差映射(F(x))的结果的维度与跳跃连接(x)的维度不同，就没有办法对它们两个进行相加操作的，必须对x进行升维操作，让他俩的维度相同时才能计算。
升维的方法有两种：
- 全0填充
- 采用1*1卷积

```

## ResNet实现

### 总体概述

本项目旨在实现ResNet模型，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、KMNIST、FashionMNIST数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN、STL-10数据集。模型最终将数据集分类为10种类别，可以根据需要增加分类数量。训练轮次默认为4轮，同样可以根据需要增加训练轮次。单通道数据集训练4~5轮就可以达到较高的精确度，而对于多通道数据，建议训练轮次在10轮以上，以增大精确度。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/ResNet)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记README文档，以及ResNet模型的模型训练和推理代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── ResNet.py    # ResNet模型训练代码
 │  └── test.py    # ResNet模型推理代码
 └── README.md 
```

### 训练及推理步骤

- 1.首先运行ResNet.py初始化ResNet网络的各参数
- 2.同时train.py会接着进行模型训练，要加载的训练数据集和测试训练集可以自己选择，本项目可以使用的数据集来源于torchvision的datasets库。相关代码如下：

```

 #下载数据集
train_set = datasets.数据集名称("下载路径",train=True,download=True,transform=pipeline)
test_set = datasets.数据集名称("下载路径",train=False,download=True,transform=pipeline)
 #加载数据集 一次性加载BATCH_SIZE个打乱顺序的数据
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

只需把数据集名称更换成你要使用的数据集(datasets中的数据集)，并修改下载数据集的位置(默认在根目录下，如果路径不存在会自动创建)即可，如果已经提前下载好了则不会下载，否则会自动下载数据集。

```

同时，程序会将每一个训练轮次的训练过程中的损失值打印出来，并给出每一轮训练的进度条以及损失值变化，损失值越接近0，则说明训练越成功。同时，每一轮训练结束后程序会打印出本轮测试的平均损失值和平均精度。

- 3.由于ResNet.py代码会将精确度最高的模型权重保存下来，以便推理的时候直接使用最好的模型，因此运行ResNet.py之前，需要设置好保存的路径，相关代码如下：

```

torch.save(model.state_dict(),'保存路径')

默认保存路径为根目录，可以根据需要自己修改路径，该文件夹不存在时程序会自动创建。

```

- 4.保存完毕后，我们可以运行test.py代码，同样需要加载数据集(和训练过程的数据相同)，步骤同2。同时，我们应将保存的最好模型权重文件加载进来，相关代码如下：

```

model.load_state_dict(torch.load("文件路径"))

文件路径为最好权重模型的路径，注意这里要写绝对路径，并且windows系统要求路径中的斜杠应为反斜杠。

```

另外，程序中创建了一个classes列表来获取分类结果，分类数量由列表中数据的数量来决定，可以根据需要来增减，相关代码如下：

```

classes=[
    "0",
    "1",
    ...
    "n-1",
]

要分成n个类别，就写0~n-1个数据项。

```

- 5.最后是推理步骤，程序会选取测试数据集的前n张图片进行推理，并打印出每张图片的预测类别和实际类别，若这两个数据相同则说明推理成功。同时，程序会将选取的图片显示在屏幕上，相关代码如下：

```

for i in range(n): #取前n张图片
    X,y=test_dataset[i][0],test_dataset[i][1]
    show(X).show()
    #把张量扩展为四维
    X=Variable(torch.unsqueeze(X, dim=0).float(),requires_grad=False).to(device)
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')

推理图片的数量即n取多少可以自己修改，但是注意要把显示出来的图片手动关掉，程序才会打印出这张图片的预测类别和实际类别。

```

## 实例

这里我们以最经典的MNIST数据集为例：

运行ResNet.py之前，要加载好要训练的数据集，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo4.png" width="50%">

以及训练好的最好模型权重best_model.pth的保存路径：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo5.png" width="50%">

这里我们设置训练轮次为4，由于没有提前下载好数据集，所以程序会自动下载在/data目录下，运行结果如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo6.png" width="50%">

最好的模型权重保存在设置好的路径中：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo7.png" width="30%">

从下图最后一轮的损失值和精确度可以看出，训练的成果已经是非常准确的了！

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo8.jpg" width="30%">

最后我们运行test.py程序，首先要把train.py运行后保存好的best_model.pth文件加载进来，设置的参数如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo9.png" width="50%">

这里我们设置推理测试数据集中的前20张图片，每推理一张图片，都会弹出来显示在屏幕上，要手动把图片关闭才能打印出预测值和实际值：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo10.png" width="30%">

由下图最终的运行结果我们可以看出，推理的结果是较为准确的，大家可以增加推理图片的数量以测试模型的准确性。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/ResNet/photo11.png" width="50%">

其他数据集的训练和推理步骤和MNIST数据集大同小异。
