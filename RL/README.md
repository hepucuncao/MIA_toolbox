# Reinforcement_learning

2024年9月1日**更新**

在此教程中，我们将对强化学习模型及其原理进行一个简单的介绍，并实现一种强化学习模型的训练和推理过程，且至少支持3种数据集，目前支持数据集有：MNIST、fashionMNIST、CIFAR10等，并给用户提供一个详细的帮助文档。

## 目录  

[基本介绍](#基本介绍)  
- [什么是强化学习](#什么是强化学习)
- [基本框架](#基本框架)
- [学习过程](#学习过程)
- [基于模型和免模型的强化学习](#基于模型和免模型的强化学习)
- [马尔科夫决策过程(MDP)](#马尔科夫决策过程(MDP))

[RL实现](#RL实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
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

### 什么是强化学习

**强化学习是一种无标签的学习，通过奖励函数来判断在确定状态下执行某一动作的好坏，学习过程就是通过奖励信号来改变执行动作的策略，最终结果就是形成一个使奖励最大的策略。**

强化学习主要由智能体(Agent)、环境(Environment)、状态(State)、动作(Action)、奖励(Reward)组成。智能体执行了某个动作后，环境将会转换到一个新的状态，对于该新的状态环境会给出奖励信号(正奖励或者负奖励)。随后，智能体根据新的状态和环境反馈的奖励，按照一定的策略执行新的动作。上述过程为智能体和环境通过状态、动作、奖励进行交互的方式。

智能体通过强化学习，可以知道自己在什么状态下，应该采取什么样的动作使得自身获得最大奖励。由于智能体与环境的交互方式与人类与环境的交互方式类似，可以认为强化学习是一套通用的学习框架，可用来解决通用人工智能的问题。因此强化学习也被称为通用人工智能的机器学习方法。简单来说，强化学习是想让一个智能体(agent)在不同的环境状态(state)下，学会选择那个使得奖赏(reward)最大的动作(action)。

### 基本框架

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo1.png" width="40%">

- 智能体(Agent)：强化学习的本体，作为学习者或者决策者，类比人的大脑;
- 环境(Environment)：智能体以外的一切，主要是状态的集合，类比人的身体以及周围的自然环境;
- 状态(state)：一个表示环境的数据，状态集则是环境中所有可能的状态;
- 动作(action)：智能体可以做出的动作，动作测试智能体可以做出的所有动作。类比人类的大脑发出的向身体发出的指令;
- 奖励(Reward)：智能体在执行一个动作后，获得的反馈信息，可以是正奖励，也可以是负奖励(惩罚);
- 策略(Policy)：环境状态到动作的映射称为策略，即智能体处在某一状态下，执行何种动作;
- 目标(Target)：强化学习的目标是自动寻找在连续时间序列里的最优策略，这里的最优策略通常指使得长期累计奖励最大化的策略，强化学习实际上是智能体在与环境进行交互的过程中，学会最佳决策序列。

### 学习过程

1.智能体感知环境状态;

2.智能体根据某种策略选择动作;

3.动作作用于环境导致环境状态变化(环境转移);

4.同时，环境向智能体发出一个反馈信号。

### 基于模型和免模型的强化学习

**(1)模型**

模型是指对环境建模，具体指状态转移概率函数和奖励函数。

**(2)基于模型的强化学习(Model-Based)**

智能体知道在任何状态下执行任何动作所获得的回报，即R(s,a)已知。可以直接使用动态规划法来求解最优策略，这种采取对环境进行建模的强化学习方法就是Model-Based强化学习。
若奖励函数和状态转移函数未知，我们就可以用特定的方法（比如神经网络或者物理机理）对它们进行模拟建模。

**(3)免模型的强化学习(Model-Free)**

其实，不需要对环境进行建模我们就可以找到最优策略，最优策略对应的是最大累计奖励，所以我们可以通过直接求解最大累计奖励，然后再根据最大累计奖励来求解最优策略，这种方法就叫做Model-Free强化学，典型的方法有Qlearning、Sarsa等

```
model-free只能在一次行动之后静静得等待现实世界给的反馈然后再取材行动，而model-base可以采用想象力预判接下来发生的所有情况，然后根据这些想象的情况选择最好的那种，并根据这种情况来采取下一步的策略。
```

### 马尔科夫决策过程(MDP)

马尔科夫性质：在时间步t+1时，环境的反馈仅取决于上一时间步t的状态s以及动作a，与时间步t-1以及t-1步之前的时间步都没有关联性。马尔科夫性是一种为了简化问题而做的假设，我们的强化学习就是基于这一假设来进行学习过程的，任何不符合该假设的问题都不太适合采用强化学习的方法来解决。马尔科夫决策过程如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo2.png" width="30%">

策略就是从大量的完整MDP序列中学习(优化)到的，在这里我们重新回到强化学习的目标：自动寻找在连续时间序列里的最优策略，这里的最优策略通常指使得长期累计奖励最大化的策略。换句话说，策略是由长期累计奖励来判断好坏的。

```
最后说一下强化学习RL与机器学习ML的区别：

RL是没有监督者的，也就是没有人告诉agent做这个动作是最好的，而是会告诉agent做这个动作能够获得多少奖励。它是一个不断试错的过程，通过很多次的尝试之后，agent慢慢修改自己的策略让自己去执行能够获得最大奖励的动作。

反馈是延时的，而不是即时。做某个动作能够获得的奖励有时并不是马上就反馈给agent的，而是有很大的时间延迟。就好像你下一步棋并不知道好坏，而是要等到最后你赢了你才知道下这一步棋是好的。或者说你做一个决定不是马上就能知道结果，而是要过很久之后，才知道当初的决定是错的还是对的。

时序是很重要的，也就是动作是一步一步来的，agent的动作会影响后续的子序列。机器学习中的数据可以看成是独立同分布的，但是在RL中，是一个动态的系统，每一步的动作会影响到后续的动作或者reward。
```

## RL实现

### 总体概述

本项目旨在实现强化学习(RL)模型，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、KMNIST、FashionMNIST等数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN等数据集。模型最终将数据集分类为10种类别，可以根据需要增加分类数量。训练轮次默认为20轮，同样可以根据需要增加训练轮次。单通道数据集训练10轮就可以达到较高的精确度，而对于多通道数据，建议训练轮次在30轮以上，以增大精确度。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/RL)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记README文档，以及RL模型的模型训练和推理代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── net.py    # RL网络模型代码
 │  ├── train.py    # RL模型训练代码
 │  └── test.py    # RL模型推理代码
 └── README.md 
```

### 训练及推理步骤

- 1.首先运行net.py初始化RL网络模型的各参数；
- 2.接着运行train.py会进行模型训练，要加载的训练数据集和测试训练集可以自己选择，本项目可以使用的数据集来源于torchvision的datasets库。相关代码如下：

```

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, dataset_name, train=True):
        if dataset_name == '数据集1':
            self.dataset = datasets.数据集1('下载路径', download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == '数据集2':
            self.dataset = datasets.数据集2('下载路径', download=True, train=train, transform=transforms.ToTensor())
        elif dataset_name == '数据集3':
            self.dataset = datasets.数据集3('下载路径', download=True, train=train, transform=transforms.ToTensor())
        else:
            raise ValueError('Unsupported dataset')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label

这里理论上可以尝试任意多种数据集，这里只写了三种，只需把数据集名称更换成你要使用的数据集(datasets中的数据集)，并修改下载数据集的位置(默认在根目录下，如果路径不存在会自动创建)即可，如果已经提前下载好了则不会下载，否则会自动下载数据集。

```

同时，程序会将每一轮训练的损失值打印出来，损失值越接近0，则说明训练越成功。同时，每一轮训练结束后程序会打印出本轮测试的平均精度。

- 3.由于train.py代码会将精确度最高的模型权重保存下来，以便推理的时候直接使用最好的模型，因此运行train.py之前，需要设置好保存的路径，相关代码如下：

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

运行train.py之前，要加载好要训练的数据集，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo3.png" width="50%">

以及训练好的最好模型权重best_model.pth的保存路径：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo4.png" width="50%">

这里我们设置训练轮次为20，由于没有提前下载好数据集，所以程序会自动下载在我们设置好的目录下，运行结果如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo5.png" width="50%">

最好的模型权重保存在设置好的路径中：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo6.png" width="30%">

从下图最后一轮的损失值和精确度可以看出，平均精度都在97~98%附近，可见训练的成果已经是非常准确的了！

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo7.png" width="30%">

最后我们运行test.py程序，首先要把train.py运行后保存好的best_model.pth文件加载进来，设置的参数如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo8.png" width="50%">

这里我们设置推理测试数据集中的前20张图片，每推理一张图片，都会弹出来显示在屏幕上，要手动把图片关闭才能打印出预测值和实际值：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo9.png" width="30%">

由下图最终的运行结果我们可以看出，推理的结果是较为准确的，大家可以增加推理图片的数量以测试模型的准确性。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo10.png" width="40%">

其他数据集的训练和推理步骤和MNIST数据集大同小异。

在上述实现的基础上，本项目又增加了一个功能代码：实现对之前训练好的模型参数生成一个哈希值。代码首先将模型的参数展平为单个向量，并初始化一串和模型等长的随机数，然后计算这个随机数和模型参数的点积，最后转换为字节表示后计算此点积的SHA-256散列值。同时，通过在代码中添加时间测量的逻辑，对模型参数打包成哈希值的时间进行了简单的评估。实现代码和结果如下所示：

```

def generate_model_hash(model):
    start_time = time.time()  #记录开始时间

    params = []
    for param in model.parameters():
        params.extend(param.detach().cpu().numpy().flatten())
    params = np.array(params)

    random_vector = np.random.rand(len(params))

    dot_product = np.dot(random_vector, params)

    dot_product_bytes = str(dot_product).encode()

    hash_value = hashlib.sha256(dot_product_bytes).hexdigest()

    end_time = time.time()  #记录结束时间
    elapsed_time = end_time - start_time  #计算耗时

    print(f"Hash generation time: {elapsed_time:.6f} seconds")  #打印耗时
    return hash_value

```

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo11.png" width="40%">

这里耗时如下图所示，可见生成哈希值的速度相对来说是很快的。注意生成哈希值的时间会受到模型参数数量和计算资源的影响，特别是在大型模型上可能会有所不同。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo20.png" width="40%">

## 成员推断攻击实现

### 总体介绍

本项目旨在实现强化学习模型的成员推断攻击，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、FashionMNIST等数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN等数据集。同时，本项目还就如何提高攻击准确率进行讨论。

<a name="MIA项目结构"></a>
### MIA项目结构

项目的目录分为两个部分：学习笔记README文档，以及RL模型的模型训练和推理代码放在MIA文件夹下。

```python
 ├── MIA    # 相关代码目录
 │  ├── classifier_method.py    # 模型训练和评估框架
 │  └── rl_model.py    # rl网络模型代码
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

2.同时可以运行rl_model.py程序以初始化强化学习模型的参数，该程序创建了一个简单的RL模型，用于执行图像分类任务。

初始化网络结构：
input_dim: 输入特征的维度。
hidden_dim: 隐藏层的神经元数量。
output_dim: 输出层的神经元数量，通常与动作空间的大小相对应。
num_layers: 隐藏层的数量(默认值为2)。
dropout_rate: dropout层的丢弃率，用于防止过拟合。

使用nn.ModuleList来存储多层全连接层(Linear)，每层后面接一个ReLU激活函数和一个dropout层。第一层将输入维度映射到隐藏维度，后续层则是隐藏层到隐藏层的映射。最后，定义一个输出层，将隐藏层的输出映射到输出维度。

forward方法定义了数据在模型中的前向传播过程：输入x被重塑为适当的形状，然后通过所有定义的层进行前向传播，最终输出结果。

get_action方法根据当前状态选择动作，状态被转换为张量，调用forward方法获得动作概率，然后使用softmax函数将输出转换为概率分布，最后选择概率最大的动作；get_action_probs方法返回当前状态下各个动作的概率，该方法与get_action类似，但不选择动作，而是返回整个概率分布。该模型可以通过get_action和get_action_probs方法来选择动作或获取动作概率，适合用于强化学习中的决策过程。

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

在classifier_methods.py程序中，定义了训练过程，接受多个参数，如模型类型('fc' 或 'rl')、隐藏层维度(fc_dim_hidden)、输入和输出维度(fc_dim_in和fc_dim_out)、批大小(batch_size)、训练轮数(epochs)、学习率(learning_rate)等。根据模型类型创建网络(FCNet/RL）)将网络移到可用的GPU/CPU。然后对训练数据和测试数据进行迭代，计算损失并更新模型参数。在训练结束时，计算并打印训练集和测试集的准确率。

```

为了提高攻击的精确度，本项目的代码在之前的成员推理攻击的代码上做了修改，包括以下几方面：
1.适配模型定义：由于给定的强化学习模型RL_model使用了不同的网络结构，因此在目标模型和影子模型的训练部分要确保适配RL_model的输入、输出和层定义。

2.优化超参数：提高攻击精确度可以通过适当调整学习率、隐藏层维度、批量大小、训练轮数等超参数来实现。同时，使用更大的影子模型数量 n_shadow（如20或更多）有助于生成更多样的攻击数据，提高攻击模型的泛化能力。

3.改进攻击数据生成方式：增加数据增强操作，比如在MNIST或其他数据集中进行随机旋转、翻转等操作，以丰富攻击模型的数据，以及平衡训练和测试集中正负样本的数量，避免数据不平衡导致的偏差。

4.改进攻击模型架构：考虑在攻击模型中使用更深的网络或带有正则化的方法（如 Dropout 或 Batch Normalization），以避免过拟合。

```
### 结果分析

本项目将以经典的数据集MNIST数据集为例，展示代码的执行过程并分析其输出结果。

首先要进行run_attack.py程序中一些参数和路径的定义，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RNN/photo17.jpg" width="50%">

全部程序运行完毕后，可以看到控制台打印出的信息，下面具体分析输出的结果。

首先是一组参数（字典）的输出，这些参数定义了模型训练的配置：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo12.png" width="100%">

其中target_model: 目标模型(RL);target_learning_rate: 目标模型的学习率;target_epochs: 目标模型训练的轮数;n_shadow: 阴影模型的数量;attack_model: 攻击模型(例如FC，全连接模型);attack_epochs: 攻击模型训练的轮数，等等。

接着开始训练目标模型，输出显示了目标模型在训练集和测试集上的准确率：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo13.png" width="50%">

开始训练阴影模型，每训练一个阴影模型(如0到9)，都会输出类似的信息，展示了该阴影模型在训练集和测试集上的准确率，并表明训练完成。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo14.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo15.png" width="50%">

训练所有阴影模型后，继续训练攻击模型，训练了针对每个类别的攻击模型，并输出每个类别的训练集和测试集准确率。同时，还会输出用于训练和测试的数据集中的样本数量，这些数字对于评估模型的性能非常重要。通常，训练集用于调整模型参数，而测试集用于评估模型在未见过的数据上的泛化能力。在理想情况下，测试集应该足够大，以便能够提供对模型性能的可靠估计，训练集也应该足够大，以便模型能够学习到数据中的模式和特征。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo16.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo17.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo18.png" width="50%">

最后打印出分类报告：输出了精确度、召回率、F1分数、支持度等指标，整体准确率在0.60~0.70附近。整体来看，修改攻击方法后准确率有所上升，但模型的表现还有提升的空间，可以进一步优化模型参数和训练策略。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo19.png" width="50%">

## 复杂场景下的成员推断攻击

### 介绍

该过程主要是在RL模型的基础之上开启复杂场景下的成员推断攻击，并添加一些新的功能代码，其中以经典数据集MNIST为例。

首先，分别对RL模型的训练数据集随机删除5%和10%的数据，记录删除了哪些数据，并分别用剩余数据重新训练RL模型，形成的模型包括原RL模型，删除5%数据后训练的RL模型，删除10%数据后训练的RL模型。然后，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和之后训练而成的模型的攻击成功率。最后，记录攻击对比情况。

上述是完全重训练的方式，即自动化地实现删除，并用剩余数据集训练一个新模型，并保存新模型和原来的模型。本文还采用了其他更快的方法，即分组重训练，具体思路为将数据分组后，假定设置被删除数据落在某一个特定组，重新训练时只需针对该组进行重训练，而不用完全重训练。同样地，保存原来模型和分组重训练后的模型。

### 代码结构
```python
 ├── Complex    # 相关代码目录
 │  ├── RL  # RL模型训练代码
 │      └── net.py    # RL网络模型代码
 │      └── rl_train.py     # RL模型完全重训练代码
 │      └── rl_part_train.py   #RL模型分组重训练代码
 ├  ├── MIA_attack  # 攻击代码
 │      └── rl_model.py    # RL网络模型代码
 │      └── fc_model.py     # FCNet神经网络模型
 │      └── run_attack.py   # 成员推断攻击代码
 ├      └── classifier_method.py    # 模型训练和评估框架
 └── README.md 
```

### 实现步骤

1. 首先进行删除数据的操作，定义一个函数remove_random_data，该函数用于从给定的PyTorch数据集中随机删除一定百分比的数据并记录被删除的数据索引，删除后返回删除的数据索引和剩余的数据索引。相关代码如下：
```

def remove_random_data(dataset, remove_percent):
    num_samples = len(dataset)
    remove_count = int(remove_percent * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    remove_indices = indices[:remove_count]
    remaining_indices = indices[remove_count:]
    return remove_indices, remaining_indices

其中，remove_percentage:要从数据集中删除的数据的百分比，remaining_indices:包含所有未被删除的数据的索引，remove_indices:被删除的数据的索引。

```
特别地，如果要使用分组重训练的方式来训练模型，删除数据的方式和上述不同。我们需要首先对训练数据集train_dataset进行分组，然后在删除数据时随机删除特定组的数据，因此再进行模型训练时我们只需要针对该组数据进行重训练，而不是从头开始完全重训练，从而加快模型训练速度。相关代码如下：

```

group_size = len(train_dataset) // n
removed_group = random.randint(0, n-1)
remaining_indices = [i for idx, i in enumerate(range(len(train_dataset))) if idx // group_size != removed_group]
remaining_train_dataset = torch.utils.data.Subset(train_dataset, remaining_indices)
train_dataloader_partial = torch.utils.data.DataLoader(remaining_train_dataset, batch_size=16, shuffle=True)

其中，n的值决定我们删除数据的比例大小，我们可以根据需要自定义地将数据分成n个组，并通过随机函数随机删除其中一个组的数据。

```

2.然后通过改变percentage的值，生成对未删除数据的数据集、随机删除5%数据后的数据集和随机删除10%数据后的数据集，然后重新训练RL模型，形成的模型包括原RL模型，删除5%数据后训练的RL模型，删除10%数据后训练的RL模型。

具体训练步骤与原来无异，区别在于要调用remove_random_data函数加载数据集并记录删除的数据，相关代码如下：
```

dataset = MyDataset(dataset_name)
remove_5_indices, remaining_5_indices = remove_random_data(dataset, 0.05)
remove_10_indices, remaining_10_indices = remove_random_data(dataset, 0.10)

注意：如果是在同一个程序中生成用不同数据集训练的模型，要记得在前一个模型训练完之后重新初始化模型，且删除5%和10%数据都是在原数据集的基础上，而不是叠加删除。

```

3.利用前面讲到的模型成员攻击算法，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和删除之后训练而成的模型的攻击成功率，并记录攻击的对比情况。

具体攻击的方法和步骤和前面讲的差不多，不同点在于，由于这里我们用的训练模型是RL模型，所以我们在rl_model.py中要构造这种模型的网络模型。

### 结果记录及分析

1.首先比较删除数据前后RL模型的训练准确率，如下图所示：

(1)完全重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo21.png" width="40%">

(图1：未删除数据的RL模型训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo22.png" width="40%">

(图2：删除5%数据后的RL训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo23.png" width="40%">

(图3：删除10%数据后的RL训练准确率)

上述三种情况的平均准确率分别为：Test Accuracy: 0.9819、Test Accuracy: 0.9832、Test Accuracy: 0.9818。

由上述结果可以看出，删除数据后模型训练的精确度先是有小幅度的升高，后急剧下降。这也说明了数据的数量和模型训练精度的关系不是线性的，它们之间存在复杂的关系，需要更多的尝试来探寻它们之间的联系，而不能一概而论！

(2)分组重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo24.png" width="50%">

(图4：将数据平均分为20组，删除5%数据后的RL训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo25.png" width="50%">

(图5：将数据平均分为10组，删除10%数据后的RL训练准确率)

训练过程中我们可以明显感觉到，采用分组重训练的方式，模型训练的速度比完全重训练快得多！这说明，使用分组重训练的方式，可以有效减少训练的时间开销。这个方法被称为增量训练，假设我们可以保存模型状态，并根据删除的数据部分，针对局部数据进行优化。

```
如果删除的数据是噪音数据或outliers，即不具代表性的数据，那么删除这些数据可能会提高模型的精确度。因为这些数据可能会干扰模型的训练，使模型学习到不正确的规律。删除这些数据后，模型可以更好地学习到数据的模式，从而提高精确度。

但是，如果删除的数据是重要的或具代表性的数据，那么删除这些数据可能会降低模型的精确度。因为这些数据可能包含重要的信息，如果删除这些数据，模型可能无法学习到这些信息，从而降低精确度。

此外，删除数据还可能会导致模型的过拟合，即模型过于拟合训练数据，无法泛化到新的数据上。这是因为删除数据后，模型可能会过于依赖剩余的数据，导致模型的泛化能力下降。
```

2.然后开始对形成的模型进行成员推理攻击，首先比较删除数据前后训练而成的RL模型的攻击成功率，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo26.png" width="50%">

(图6：未删除数据的RL模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo27.png" width="50%">

(图7：删除5%数据后的RL模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/RL/photo28.png" width="50%">

(图8：删除10%数据后的RL模型攻击成功率)

由上述结果可知，随着删除数据的比例增加，模型成员推断攻击的成功率先是有细微地升高，然后又有细微地降低，但删除10%数据后的RL模型攻击成功率跟不删除数据时的攻击成功率是差不多的。

```
删除一部分数据再进行模型成员推断攻击，攻击的成功率可能会降低。这是因为模型成员推断攻击的原理是利用模型对训练数据的记忆，通过观察模型对输入数据的行为来判断该数据是否在模型的训练集中。

如果删除了一部分数据，模型的训练集就会减少，模型对剩余数据的记忆就会减弱。这样，攻击者就更难以通过观察模型的行为来判断某个数据是否在模型的训练集中，从而降低攻击的成功率。

此外，删除数据还可能会使模型变得更robust，对抗攻击的能力更强。因为模型在训练时需要适应新的数据分布，模型的泛化能力就会提高，从而使攻击者更难以成功地进行成员推断攻击。

但是，需要注意的是，如果删除的数据是攻击者已经知晓的数据，那么攻击的成功率可能不会降低。因为攻击者已经知道这些数据的信息，仍然可以使用这些信息来进行攻击。

本项目所采用的模型都是神经网络类的，如果采用非神经网络类的模型，例如，决策树、K-means等，可能会有不一样的攻击效果，读者可以尝试一下更多类型的模型观察一下。
```

同样地，我们在删除数据后重新训练模型时，也添加了一个功能代码，上述实现的基础上，实现对之前训练好的模型参数生成一个哈希值。代码首先将模型的参数展平为单个向量，并初始化一串和模型等长的随机数，然后计算这个随机数和模型参数的点积，最后转换为字节表示后计算此点积的SHA-256散列值。同时，通过在代码中添加时间测量的逻辑，对模型参数打包成哈希值的时间进行了简单的评估。实现代码和结果如下所示：
```

def compute_model_hash(model):
    # 获取模型参数
    model_params = torch.cat([p.view(-1) for p in model.parameters()])
    # 初始化随机数
    random_tensor = torch.randn_like(model_params)
    # 计算点积
    dot_product = torch.dot(model_params, random_tensor)
    # 开始计时
    start_time = time.time()
    # 计算 SHA-256 哈希值
    hash_value = hashlib.sha256(dot_product.detach().cpu().numpy()).hexdigest()
    # 计算耗时
    elapsed_time = time.time() - start_time
    return hash_value, elapsed_time

```

未删除数据、删除5%数据、删除10%数据时的哈希值及耗时分别为：

Model Hash: 26a6d963c2f866230324150843d4db23551e6153dca0688e53f8b065dbdc430c, Time taken: 0.000013 seconds

Model Hash (5% removed): 086c683051aaa6fdd2d9c25463ed77717176d87ab8dc58153042f3e87fcfd3fb, Time taken: 0.0000009 seconds

Model Hash (10% removed): af41c7acb2acb8f10008c870fe4924e5d07e0ffa00d756bc8741e8da766e2d9e, Time taken: 0.000008 seconds

