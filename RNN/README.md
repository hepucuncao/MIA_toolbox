# 循环神经网络(RNN)

2024年7月20日**更新**

在此教程中，我们将对循环神经网络RNN模型及其原理进行一个简单的介绍，并实现RNN模型的训练和推理，目前支持MNIST、FashionMNIST和CIFAR-10等数据集，并给用户提供一个详细的帮助文档。同时，本项目还将实现循环神经网络的模型成员推理攻击，以及复杂场景下的成员推理攻击。

## 目录  

[基本介绍](#基本介绍)  
- [RNN总体概述](#RNN总体概述)
- [什么是RNNs?](#什么是RNNs？)
- [RNNs的应用](#RNNs的应用)
- [如何训练RNNs](#如何训练RNNs)
- [RNNs扩展和改进模型](#RNNs扩展和改进模型)

[RNN实现](#RNN实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
- [RNN模型构造分析](#RNN模型构造分析)
- [训练及推理步骤](#训练及推理步骤)
- [实例](#实例)

[成员推断攻击实现](#成员推断攻击实现)
- [总体介绍](#总体介绍)
- [MIA项目结构](#项目结构)
- [实现步骤及分析](#实现步骤及分析)
- [结果分析](#结果分析)

[复杂场景下的成员推断攻击](#复杂场景下的成员推断攻击)
- [介绍](#介绍)
- [代码结构](#代码结构)
- [实现步骤](#实现步骤)
- [结果记录及分析](#结果记录及分析)

## 基本介绍

### RNN总体概述

循环神经网络(RNNs)是神经网络中一个大家族，它们主要用于文本、信号等序列相关的数据。不同于传统的FNNs(Feed-forward Neural Networks，前向反馈神经网络)，RNNs引入了定向循环，能够处理那些输入之间前后关联的问题。定向循环结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo1.png" width="50%">


### 什么是RNNs？

RNNs的目的是用来处理序列数据。在传统的神经网络模型中，是从输入层到隐含层再到输出层，层与层之间是全连接的，每层之间的节点是无连接的，但是这种普通的神经网络对于很多问题却无能无力。例如，你要预测句子的下一个单词是什么，一般需要用到前面的单词，因为一个句子中前后单词并不是独立的。RNNs之所以称为循环神经网络，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。理论上，RNNs能够对任何长度的序列数据进行处理。但是在实践中，为了降低复杂性往往假设当前的状态只与前面的几个状态相关，下图便是一个典型的RNNs：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo2.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo3.png" width="50%">


RNNs包含输入单元(Input units)，输入集标记为{x0,x1,...,xt,xt+1,...}，而输出单元(Output units)的输出集则被标记为{y1,y2,...,yt,yt+1,...}，RNNs还包含隐藏单元(Hidden units)，我们将其输出集标记为{s1,s2,...,st,st+1,...}，这些隐藏单元完成了最为主要的工作，它是网络的记忆单元。为了降低网络的复杂度，往往st只包含前面若干步而不是所有步的隐藏层状态。如上图所示，有一条单向流动的信息流是从输入单元到达隐藏单元的，与此同时另一条单向流动的信息流从隐藏单元到达输出单元。在某些情况下，RNNs会打破后者的限制，引导信息从输出单元返回隐藏单元(称为“Back Projections)，并且隐藏层的输入还包括上一隐藏层的状态，即隐藏层内的节点可以自连也可以互连。

上图将循环神经网络进行展开成一个全神经网络。例如，对一个包含5个单词的语句，那么展开的网络便是一个五层的神经网络，每一层代表一个单词。

```
注意：使用计算机对自然语言进行处理，便需要将自然语言处理成为机器能够识别的符号，加上在机器学习过程中，需要将其进行数值化。词是自然语言理解与处理的基础，因此需要对词进行数值化，词向量(Word Representation，Word embeding)是一种可行又有效的方法，即使用一个指定长度的实数向量v来表示一个词。最简单的表示方法，就是使用One-hot vector表示单词，即根据单词的数量|V|生成一个|V| * 1的向量，当某一位为1的时候其他位都为零，这个向量就代表一个单词。
```

在传统神经网络中，每一个网络层的参数是不共享的。而在RNNs中，每输入一步，每一层各自都共享参数U,V,W，反映了RNNs中的每一步都在做相同的事，只是输入不同，因此大大地降低了网络中需要学习的参数。具体来说，传统神经网络的参数是不共享的，并不是表示对于每个输入有不同的参数，而是将RNN是进行展开，这样变成了多层的网络，如果这是一个多层的传统神经网络，那么xt到st之间的U矩阵与xt+1到st+1之间的U矩阵是不同的，而RNN中却是一样的，同理对于s与s层之间的W、s层与o层之间的V也是一样的。

上图中每一步都会有输出，但是每一步都要有输出并不是必须的。同理，每步都需要输入也不是必须的。RNNs的关键之处在于隐藏层，隐藏层能够捕捉序列的信息。

### RNNs的应用

RNNs已经被在实践中证明对NLP是非常成功的，在RNNs中，目前使用最广泛最成功的模型是LSTMs(Long Short-Term Memory，长短时记忆模型)模型，该模型通常比vanilla RNNs能够更好地对长短时依赖进行表达，相对于一般的RNNs，该模型只是在隐藏层做了手脚。下面对RNNs在NLP中的应用进行简单的介绍：

**语言模型与文本生成(Language Modeling and Generating Text)**

 给出一个单词序列，我们需要根据前面的单词预测每一个单词的可能性。语言模型能够一个语句正确的可能性，这是机器翻译的一部分，往往可能性越大，语句越正确。另一种应用便是使用生成模型预测下一个单词的概率，从而生成新的文本根据输出概率的采样。语言模型中，典型的输入是单词序列中每个单词的词向量(如 One-hot vector)，输出时预测的单词序列。当在对网络进行训练时，如果ot=xt+1，那么第t步的输出便是下一步的输入。

**机器翻译(Machine Translation)**

机器翻译是将一种源语言语句变成意思相同的另一种源语言语句，如将英语语句变成同样意思的中文语句。与语言模型关键的区别在于，需要将源语言语句序列输入后，才进行输出，即输出第一个单词时，便需要从完整的输入序列中进行获取。机器翻译如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo4.png" width="60%">

**语音识别(Speech Recognition)**

语音识别是指给一段声波的声音信号，预测该声波对应的某种指定源语言的语句以及该语句的概率值。

**图像描述生成 (Generating Image Descriptions)**

和卷积神经网络(convolutional Neural Networks, CNNs)一样，RNNs已经在对无标图像描述自动生成中得到应用。将CNNs与RNNs结合进行图像描述自动生成，该组合模型能够根据图像的特征生成描述。如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo5.png" width="60%">

### 如何训练RNNs

对于RNNs的训练和对传统的ANN训练一样，同样使用BP误差反向传播算法，区别在于，如果将RNNs进行网络展开，那么参数W,U,V是共享的，而传统神经网络不是。并且在使用梯度下降算法中，每一步的输出不仅依赖当前步的网络，并且还以来前面若干步网络的状态。该学习算法称为Backpropagation Through Time (BPTT)。需要注意的是，在vanilla RNNs训练中，BPTT无法解决长时依赖问题(即当前的输出与前面很长的一段序列有关，一般超过十步就无能为力了)，因为BPTT会带来所谓的梯度消失或梯度爆炸问题(the vanishing/exploding gradient problem)。

```
在Pytorch中，需要学习的参数只有U、W、b_u、b_w四个，并没有V，所以它是把某一状态的隐藏状态h_t直接作为输出，没有与参数V相乘的这个过程，并且RNN输入序列长度和输出序列长度是完全相同的。

nn.RNN()方法中，有以下几个参数：

 1.input_size：输入数据的特征数量。

 2.hidden_size：隐藏层h的特征数量，实际上也是输出数据的维度。

 3.bidirectional：是否为双向RNN，默认False。

 4.num_layers：RNN层数，默认为1。多层即为深层RNN。

 5.batch_firse：是否认为输入数据第一个维度为batch_size，默认为True。如为False，则认为第一个维度是序列长度。

 6.nonlinearity：非线性函数，可以为'tanh'、'relu'、'sigmoid'等激活函数。

 7.bias：参数中是否有偏置b，默认为True。

 nn.RNN()最终的输出并只有output，还有最后一个隐藏状态h_t。所以它的输出是(output，h_t)元组。对于RNNCell也是如此。

对于nn.RNNCell，参数与RNN大致相同。不过RNNCell是RNN一个结构单元，接受序列长度只能为1，需要用一定方法把它们组合才能正常使用。
```
### RNNs扩展和改进模型

**Simple RNNs(SRNs)**

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo6.png" width="20%">

SRNs是RNNs的一种特例，它是一个三层网络，并且在隐藏层增加了上下文单元，上图中的便是隐藏层，u便是上下文单元。上下文单元节点与隐藏层中的节点的连接是固定的，并且权值也是固定的，即一个上下文节点与隐藏层节点一一对应，并且值是确定的。在每一步中，使用标准的前向反馈进行传播，并使用学习算法进行学习。上下文每一个节点保存其连接的隐藏层节点的上一步的输出，即保存上文，并作用于当前步对应的隐藏层节点的状态，即隐藏层的输入由输入层的输出与上一步的自己的状态所决定的。因此SRNs能够解决标准的多层感知机(MLP)无法解决的对序列数据进行预测的任务。

**Bidirectional RNNs**

Bidirectional RNNs(双向网络)的改进之处便是，假设当前的输出(第t步的输出)不仅与前面的序列有关，并且还与后面的序列有关。Bidirectional RNNs是一个相对较简单的RNNs，是由两个RNNs上下叠加在一起组成的。输出由这两个RNNs的隐藏层的状态决定的。如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo7.png" width="30%">

**Deep(Bidirectional)RNNs**

Deep(Bidirectional)RNNs与Bidirectional RNNs相似，只是对于每一步的输入有多层网络，使该网络便有更强大的表达与学习能力，但复杂性也提高了，同时需要更多的训练数据。Deep(Bidirectional)RNNs的结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo8.png" width="30%">

**Gated Recurrent Unit Recurrent Neural Networks**

GRUs是一般的RNNs的改良版本，主要是从以下两个方面进行改进。一、序列中不同的位置处的单词对当前的隐藏层的状态的影响不同，越前面的影响越小，即每个前面状态对当前的影响进行了距离加权，距离越远，权值越小。二、在产生误差error时，误差可能是由某一个或者几个单词而引发的，所以应仅仅对对应的单词weight进行更新。GRUs的结构如下图所示。GRUs首先根据当前输入单词向量word vector以及前一个隐藏层的状态hidden state计算出update gate和reset gate，再根据reset gate、当前word vector以及前一个hidden state计算新的记忆单元内容(new memory content)。当reset gate为1的时候，new memory content忽略之前的所有memory content，最终的memory是之前的hidden state与new memory content的结合。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo9.png" width="60%">

**LSTM Netwoorks**

LSTMs与GRUs类似，它与一般的RNNs结构本质上并没有什么不同，只是使用了不同的函数去去计算隐藏层的状态。在LSTMs中，i结构被称为cells，可以把cells看作是黑盒用以保存当前输入xt之前的保存的状态ht−1，这些cells决定哪些cell抑制哪些cell兴奋，结合前面的状态、当前的记忆与当前的输入。已经证明，该网络结构在对长序列依赖问题中非常有效。LSTMs的网络结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo10.png" width="60%">

**Clockwork RNNs(CW-RNNs)**

CW-RNNs也是一个RNNs的改良版本，是一种使用时钟频率来驱动的RNNs。它将隐藏层分为几个块(组，Group/Module)，每一组按照自己规定的时钟频率对输入进行处理。并且为了降低标准的RNNs的复杂性，CW-RNNs减少了参数的数目，提高了网络性能，加速了网络的训练。CW-RNNs通过不同的隐藏层模块工作在不同的时钟频率下来解决长时间依赖问题。将时钟时间进行离散化，然后在不同的时间点，不同的隐藏层组在工作。因此，所有的隐藏层组在每一步不会都同时工作，这样便会加快网络的训练。并且，时钟周期小的组的神经元的不会连接到时钟周期大的组的神经元，只会周期大的连接到周期小的(即组与组之间的连接是有向的，代表信息的传递是有向的)，周期大的速度慢，周期小的速度快，那么便是速度慢的连接速度快的，反之则不成立。CW-RNNs的网络结构如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo11.png" width="60%">

## RNN实现

### 总体概述

本项目旨在实现RNN模型，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、KMNIST、FashionMNIST数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN、STL-10数据集。模型最终将数据集分类为10种类别，可以根据需要增加分类数量。训练轮次默认为20轮，同样可以根据需要增加训练轮次。单通道数据集训练20轮就可以达到较高的精确度，而对于多通道数据，建议增大训练轮次以提高精确度，但是过多的话会出现过拟合的状况，导致精度不会明显增加。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/RNN)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记README文档，以及ResNet模型的模型训练和推理代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── net.py    # RNN模型代码
 │  └── train.py    # RNN模型训练代码
 │  └── test.py    # RNN模型推理代码
 └── README.md 
```

### RNN模型构造分析

代码定义了一个基于LSTM(长短期记忆网络)的循环神经网络(RNN)模型，下面是具体的分析：

```
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
```

RNN 继承自 nn.Module，这是所有神经网络模块的基类。其中，input_size：输入特征的维度；hidden_size：LSTM隐藏层的维度；num_layers：LSTM层的数量；num_classes：输出类别的数量。

self.lstm：创建一个LSTM层，batch_first=True表示输入的张量形状为 (batch_size, seq_length, input_size)。

self.fc：创建一个全连接层，将LSTM的输出映射到目标类别。

```def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0)) 
    out = self.fc(out[:, -1, :])
    return out
```

forward方法定义了前向传播的过程。h0 和 c0初始化LSTM的隐藏状态和细胞状态，形状为(num_layers, batch_size, hidden_size)，并将其移动到指定的设备(GPU或CPU)；self.lstm(x, (h0, c0))：将输入x和初始状态传入LSTM，返回的out是LSTM的输出，形状为(batch_size, seq_length, hidden_size)；out[:, -1, :]提取LSTM最后一个时间步的输出；self.fc(out[:, -1, :])将最后一个时间步的输出传递给全连接层，得到最终的输出。

### 推理及训练步骤

- 1.首先运行net.py初始化RNN网络模型的各个参数
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
    # 把张量扩展为三维
    X = Variable(X.float().view(1, 28, -1), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')

推理图片的数量即n取多少可以自己修改，注意要把显示出来的图片手动关掉，程序才会打印出这张图片的预测类别和实际类别。

注意：和之前实现的卷积神经网络不同的是，这里要把张量拓展为三维，而不是之前的思维。

```

### 实例

这里我们以最经典的MNIST数据集为例：

成功运行完net.py程序后，加载train.py程序的数据集：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo13.png" width="50%">

以及best_model.pth的保存路径：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo12.png" width="50%">

这里我们设置训练轮次为20，由于没有提前下载好数据集，所以程序会自动下载在/MNIST目录下，运行结果如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo12.png" width="50%">

最好的模型权重保存在设置好的路径中：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo13.png" width="30%">

从下图最后一轮的损失值和精确度可以看出，训练的成果已经是非常准确的了，几乎保持在98、99%附近。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo14.png" width="30%">

最后我们运行test.py程序，首先要把train.py运行后保存好的best_model.pth文件加载进来，设置的参数如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo15.png" width="50%">

这里我们设置推理测试数据集中的前20张图片，每推理一张图片，都会弹出来显示在屏幕上，要手动把图片关闭才能打印出预测值和实际值：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo15.png" width="30%">

由下图最终的运行结果我们可以看出，推理的结果是较为准确的，大家可以增加推理图片的数量以测试模型的准确性。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo16.png" width="50%">

其他数据集的训练和推理步骤和MNIST数据集大同小异。

## 成员推断攻击实现

### 总体介绍

本项目旨在实现循环神经网络模型的成员推断攻击，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、FashionMNIST等数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN等数据集。

<a name="MIA项目结构"></a>
### MIA项目结构

项目的目录分为两个部分：学习笔记README文档，以及RNN模型的模型训练和推理代码放在train文件夹下。

```python
 ├── MIA    # 相关代码目录
 │  ├── classifier_method.py    # 模型训练和评估框架
 │  └── rnn_model.py    # rnn网络模型代码
 │  └── fc_model.py     # FCNet神经网络模型
 │  └── run_attack.py   #成员推断攻击代码
 └── README.md 
```

### 实现步骤及分析

1.首先运行fc_model.py程序以初始化FCNet神经网络模型的参数，该程序定义了一个简单的全连接神经网络模型，包括一个隐藏层和一个输出层，用于将输入数据映射到指定的输出维度。在前向传播过程中，通过激活函数ReLU实现非线性变换。

```
输入参数包括dim_in(输入维度，默认为10)、dim_hidden(隐藏层维度，默认为20)、dim_out(输出维度，默认为2)、batch_size(批处理大小，默认为100)和rtn_layer(是否返回隐藏层输出，默认为True)。

然后定义了两个全连接层(fc1和fc2)，分别将输入维度dim_in映射到隐藏层维度dim_hidden，将隐藏层映射到输出维度dim_out。

forward函数定义了数据在模型中的前向传播过程。输入x经过第一个全连接层fc1后，通过激活函数ReLU进行非线性变换，然后再经过第二个全连接层fc2得到输出。
```

2.同时可以运行rnn_model.py程序以初始化RNN循环神经网络模型的参数，该程序创建了一个简单的RNN模型，用于执行图像分类任务。模型使用LSTM（长短期记忆网络）作为其核心组件。

初始化网络结构：
- height和width：输入数据的宽度和高度；
- hidden_size：LSTM隐藏层的大小；
- output_size：输出层的大小；
- input_size：计算输入大小，等于height * width；
- self.lstm：定义一个LSTM层，输入大小为input_size，隐藏层大小为hidden_size，batch_first=True表示输入和输出的第一个维度是batch大小；
- self.fc：定义一个全连接层（线性层），将LSTM的输出映射到output_size。

forward方法定义了数据在模型中的前向传播过程：

首先，定义x.view(x.size(0), x.size(1), -1)将输入x重塑为(batch_size, seq_length, input_size)的形状，其中-1表示自动计算该维度的大小；接着定义self.lstm(x)将重塑后的输入传递给LSTM层，返回输出out和隐藏状态（这里用_忽略隐藏状态）；最后用self.fc(out[:, -1, :])取LSTM输出的最后一个时间步的输出，并通过全连接层，得到最终的输出。

```
注意：如果网络接受灰度图像而不是彩色图像，conv1的滤波器通道数的注释应从3更改为1，同样，fc1层的输入维度是根据conv2的输出展平后的结果计算的。
```

3.接着运行run.attack.py程序，其中会调用classifier_methods.py程序。代码主要实现了一个攻击模型的训练过程，包括目标模型、阴影模型和攻击模型的训练，可以根据给定的参数设置进行模型训练和评估。

运行代码之前，要先定义一些常量和路径，包括训练集和测试集的大小、模型保存路径、数据集路径等，数据集若未提前下载程序会自动下载，相关代码如下：

```
TRAIN_SIZE = 10000
TEST_SIZE = 500

TRAIN_EXAMPLES_AVAILABLE = 50000
TEST_EXAMPLES_AVAILABLE = 10000

MODEL_PATH = '模型保存路径'
DATA_PATH = '数据保存路径'

trainset = torchvision.datasets.数据集名称(root='保存路径', train=True, download=True,
                                            transform=transform)

testset = torchvision.datasets.数据集名称(root='保存路径', train=False, download=True,
                                           transform=transform)

if save:
torch.save((attack_x, attack_y, classes), MODEL_PATH + '参数文件名称')

```

其中，full_attack_training函数实现了完整的攻击模型训练过程，包括训练目标模型、阴影模型和攻击模型。在训练目标模型时，会根据给定的参数设置构建数据加载器，训练模型并保存用于攻击模型的数据。在训练阴影模型时，会循环训练多个阴影模型，并保存用于攻击模型的数据。最后，在训练攻击模型时，会根据目标模型和阴影模型的数据进行训练，评估攻击模型的准确率和生成分类报告。

train_target_model和train_shadow_models函数分别用于训练目标模型和阴影模型，包括数据准备、模型训练和数据保存等操作；train_attack_model函数用于训练攻击模型，包括训练不同类别的攻击模型、计算准确率和生成分类报告等操作。

在classifier_methods.py程序中，定义了训练过程，接受多个参数，如模型类型('fc' 或 'rnn')、隐藏层维度(fc_dim_hidden)、输入和输出维度(fc_dim_in 和 fc_dim_out)、批大小(batch_size)、训练轮数(epochs)、学习率(learning_rate)等。根据模型类型创建网络(FCNet/RNN）)将网络移到可用的GPU/CPU。然后对训练数据和测试数据进行迭代，计算损失并更新模型参数。在训练结束时，计算并打印训练集和测试集的准确率。

### 结果分析

本项目将以经典的数据集MNIST数据集为例，展示代码的执行过程并分析其输出结果。

首先要进行run_attack.py程序中一些参数和路径的定义，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo17.jpg" width="50%">

全部程序运行完毕后，可以看到控制台打印出的信息，下面具体分析输出的结果。

首先是一组参数（字典）的输出，这些参数定义了模型训练的配置：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo18.png" width="100%">

其中target_model: 目标模型(例如RNN);target_learning_rate: 目标模型的学习率;target_epochs: 目标模型训练的轮数;n_shadow: 阴影模型的数量;attack_model: 攻击模型(例如FC，全连接模型);attack_epochs: 攻击模型训练的轮数，等等。

接着开始训练目标模型，输出显示了目标模型在训练集和测试集上的准确率：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo19.png" width="50%">

开始训练阴影模型，每训练一个阴影模型(如0到9)，都会输出类似的信息，展示了该阴影模型在训练集和测试集上的准确率，并表明训练完成。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo20.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo21.png" width="50%">

训练所有阴影模型后，继续训练攻击模型，训练了针对每个类别的攻击模型，并输出每个类别的训练集和测试集准确率。同时，还会输出用于训练和测试的数据集中的样本数量，这些数字对于评估模型的性能非常重要。通常，训练集用于调整模型参数，而测试集用于评估模型在未见过的数据上的泛化能力。在理想情况下，测试集应该足够大，以便能够提供对模型性能的可靠估计，训练集也应该足够大，以便模型能够学习到数据中的模式和特征。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo22.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo23.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo24.png" width="50%">

最后打印出分类报告：输出了精确度、召回率、F1分数、支持度等指标，整体准确率在0.50~0.60附近。整体来看，模型的表现还有提升的空间，可以进一步优化模型参数和训练策略。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo25.png" width="50%">

## 复杂场景下的成员推断攻击

### 介绍

该过程主要是在RNN模型的基础之上开启复杂场景下的成员推断攻击，以经典数据集MNIST为例。

首先，分别对RNN模型的训练数据集随机删除5%和10%的数据，记录删除了哪些数据，并分别用剩余数据重新训练RNN模型，形成的模型包括原RNN模型，删除5%数据后训练的RNN模型，删除10%数据后训练的RNN模型。然后，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和之后训练而成的模型的攻击成功率。最后，记录攻击对比情况。

上述是完全重训练的方式，即自动化地实现删除，并用剩余数据集训练一个新模型，并保存新模型和原来的模型。本文还采用了其他更快的方法，即分组重训练，具体思路为将数据分组后，假定设置被删除数据落在某一个特定组，重新训练时只需针对该组进行重训练，而不用完全重训练。同样地，保存原来模型和分组重训练后的模型。

### 代码结构
```python
 ├── Complex    # 相关代码目录
 │  ├── RNN   # RNN模型训练代码
 │      └── net.py    # RNN网络模型代码
 │      └── rnn_train.py     # RNN模型完全重训练代码
 │      └── rnn_part_train.py   #RNN模型分组重训练代码
 ├  ├── MIA_attack  # 攻击代码
 │      └── rnn_model.py    # RNN网络模型代码
 │      └── fc_model.py     # FCNet神经网络模型
 │      └── run_attack.py   # 成员推断攻击代码
 ├      └── classifier_method.py    # 模型训练和评估框架
 └── README.md 
```

### 实现步骤

1. 首先进行删除数据的操作，定义一个函数remove_data，该函数用于从给定的PyTorch数据集中随机删除一定百分比的数据，并返回剩余的数据集和被删除的数据的索引。相关代码如下：
```

def remove_data(dataset, percentage):
    indices = list(range(len(dataset)))
    num_to_remove = int(len(dataset) * percentage)
    removed_indices = random.sample(indices, num_to_remove)
    remaining_indices = [i for i in indices if i not in removed_indices]
    remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)
    return remaining_dataset, removed_indices

其中，percentage:要从数据集中删除的数据的百分比，remaining_indices:包含所有未被删除的数据的索引，remaining_dataset:剩余的数据集，removed_indices:被删除的数据的索引。

```
特别地，如果要使用分组重训练的方式来训练模型，删除数据的方式和上述不同。我们需要首先对训练数据集train_dataset进行分组，然后在删除数据时随机删除特定组的数据，因此再进行模型训练时我们只需要针对该组数据进行重训练，从而加快模型训练速度。相关代码如下：

```

group_size = len(train_dataset) // n
removed_group = random.randint(0, n-1)
remaining_indices = [i for idx, i in enumerate(range(len(train_dataset))) if idx // group_size != removed_group]
remaining_train_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
train_dataloader_partial = torch.utils.data.DataLoader(remaining_train_dataset, batch_size=16, shuffle=True)

其中，n的值决定我们删除数据的比例大小，我们可以根据需要自定义地将数据分成n个组，并通过随机函数随机删除其中一个组的数据。

```

2.然后通过改变percentage的值，生成对未删除数据的数据集、随机删除5%数据后的数据集和随机删除10%数据后的数据集，然后重新训练RNN模型，形成的模型包括原RNN模型，删除5%数据后训练的RNN模型，删除10%数据后训练的RNN模型。

具体训练步骤与原来无异，区别在于要调用remove_data函数生成删除数据后的数据集，举例如下：
```

remaining_train_dataset_5, removed_indices_5 = remove_data(train_dataset, 0.05)
train_dataloader_5 = torch.utils.data.DataLoader(remaining_train_dataset_5, batch_size=16, shuffle=True)

注意：如果是在同一个程序中生成用不同数据集训练的模型，要记得在前一个模型训练完之后重新初始化模型，如model = SimpleRNN(height, width, hidden_size, output_size).to(device)，且删除5%和10%数据都是在原数据集的基础上，而不是叠加删除。

```

3.利用前面讲到的模型成员攻击算法，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和删除之后训练而成的模型的攻击成功率，并记录攻击的对比情况。

具体攻击的方法和步骤和前面讲的差不多，不同点在于，由于这里我们用的训练模型是RNN模型，所以我们在rnn_model.py中要构造这种模型的网络模型。

### 结果记录及分析

1.首先比较删除数据前后RNN模型的训练准确率，如下图所示：

(1)完全重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo26.png" width="30%">

(图1：未删除数据的RNN模型训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo27.png" width="30%">

(图2：删除5%数据后的RNN训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo28.png" width="30%">

(图3：删除10%数据后的RNN训练准确率)

由上述结果可以看出，删除数据后模型训练的精确度先是有小幅度的升高，后急剧下降。这也说明了数据的数量和模型训练精度的关系不是线性的，它们之间存在复杂的关系，需要更多的尝试来探寻它们之间的联系，而不能一概而论！

(2)分组重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo29.png" width="30%">

(图4：删除5%数据后的RNN训练准确率，这里随机删除了第1组的数据)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo30.png" width="30%">

(图5：删除10%数据后的RNN训练准确率，这里随机删除了3组的数据)

训练过程中我们可以明显感觉到，采用分组重训练的方式，模型训练的速度比完全重训练快得多！

```
如果删除的数据是噪音数据或outliers，即不具代表性的数据，那么删除这些数据可能会提高模型的精确度。因为这些数据可能会干扰模型的训练，使模型学习到不正确的规律。删除这些数据后，模型可以更好地学习到数据的模式，从而提高精确度。

但是，如果删除的数据是重要的或具代表性的数据，那么删除这些数据可能会降低模型的精确度。因为这些数据可能包含重要的信息，如果删除这些数据，模型可能无法学习到这些信息，从而降低精确度。

此外，删除数据还可能会导致模型的过拟合，即模型过于拟合训练数据，无法泛化到新的数据上。这是因为删除数据后，模型可能会过于依赖剩余的数据，导致模型的泛化能力下降。
```

2.然后开始对形成的模型进行成员推理攻击，首先比较删除数据前后训练而成的RNN模型的攻击成功率，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo31.png" width="30%">

(图6：未删除数据的RNN模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo32.png" width="30%">

(图7：删除5%数据后的RNN模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo33.png" width="30%">

(图8：删除10%数据后的RNN模型攻击成功率)

由上述结果可知，随着删除数据的比例增加，模型成员推断攻击的成功率先是有细微地升高，然后又有细微地降低，但删除10%数据后的RNN模型攻击成功率跟不删除数据时的攻击成功率是差不多的。

```
删除一部分数据再进行模型成员推断攻击，攻击的成功率可能会降低。这是因为模型成员推断攻击的原理是利用模型对训练数据的记忆，通过观察模型对输入数据的行为来判断该数据是否在模型的训练集中。

如果删除了一部分数据，模型的训练集就会减少，模型对剩余数据的记忆就会减弱。这样，攻击者就更难以通过观察模型的行为来判断某个数据是否在模型的训练集中，从而降低攻击的成功率。

此外，删除数据还可能会使模型变得更robust，对抗攻击的能力更强。因为模型在训练时需要适应新的数据分布，模型的泛化能力就会提高，从而使攻击者更难以成功地进行成员推断攻击。

但是，需要注意的是，如果删除的数据是攻击者已经知晓的数据，那么攻击的成功率可能不会降低。因为攻击者已经知道这些数据的信息，仍然可以使用这些信息来进行攻击。

本项目所采用的模型都是神经网络类的，如果采用非神经网络类的模型，例如，决策树、K-means等，可能会有不一样的攻击效果，读者可以尝试一下更多类型的模型观察一下。
```

由结果可知，此时对于RNN模型的成员推理攻击准确率在50%上下波动，这和盲猜的概率是差不多的，因此下面我们分析可以通过什么方式来提高成员推理攻击的准确率。

提高成员推理攻击MIA准确率的关键在于改进训练模型的方式、数据平衡以及特征提取方式。这里有一些优化方向可以尝试：

1.改进目标模型和影子模型的训练过程：

eg.增加模型复杂性：增加隐藏层维度、使用更深的网络或更复杂的网络结构（如LSTM代替简单的RNN），可以提高模型的表现。
增加训练数据量和训练轮次：提高目标和影子模型的训练数据量以及训练轮数可以让模型更好地拟合数据。

2.优化攻击模型的训练过程：

eg.数据平衡：在攻击模型训练时，正类（被成员推理为训练数据）和负类（被推理为非训练数据）的数量通常是不平衡的，因此平衡数据可以提高攻击模型的泛化能力。
特征选择：将中间层的特征（如激活值）作为攻击模型的输入比直接使用输出层的概率分布更有可能提升攻击的准确率。
尝试不同的攻击模型架构：

3.使用更复杂的攻击模型：上述攻击模型是一个简单的全连接网络，可以尝试使用更复杂的架构来捕捉更复杂的输入模式。

参考文献：On the Privacy Risks of Deploying Recurrent Neural Networks in Machine Learning Models （https://arxiv.org/pdf/2110.03054）

改进后的代码参考Complex-Improve-improve.py

进行上述改进后，可以看到，针对RNN模型的成员推理攻击的准确率已经有了小幅提高。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo34.png" width="30%">
