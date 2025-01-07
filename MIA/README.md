# Membership Inference Attack

2024年5月12日**更新**

在此教程中，我们将对成员推断攻击的定义及其原理进行一个简单的介绍，并实现成员推断攻击模型，目前支持数据集有：MNIST、fashionMNIST、CIFAR10等，同时给用户提供一个详细的说明和帮助文档。

## 目录  

[基本介绍](#基本介绍)  
- [定义](#定义)
- [核心思想](#核心思想)
- [分类](#分类)

[影子模型](#影子模型)
- [影子模型概述](#影子模型概述)
- [生成影子模型的训练数据](#生成影子模型的训练数据)
- [训练攻击模型](#训练攻击模型)
- [评价](#评价)

[成员推断攻击实现](#成员推断攻击实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
- [实现步骤及分析](#实现步骤及分析)
- [结果分析](#结果分析)

[复杂场景下的成员推断攻击](#复杂场景下的成员推断攻击)
- [总体介绍](#总体介绍)
- [代码结构](#代码结构)
- [实现步骤](#实现步骤)
- [结果记录及分析](#结果记录及分析)

## 基本介绍

### 定义

关于成员推理攻击的目的，或者是说他的定义，就是为了分辨出某些数据样本是否被用于某一机器学习模型的训练过程。换句话来讲，对于攻击者来说这就是一个二分类任务，对于这一方向的研究就是使用不同的tricks来解决这个二分类问题。

成员推理攻击利用了这样一种观察，即机器学习模型在它们所训练的数据上的行为常常与它们第一次“看到”的数据不同。过拟合是一个常见的原因，但不是唯一的原因。攻击者的目的是构建一个攻击模型，该模型可以识别目标模型行为中的这些差异，并利用它们来区分目标模型的成员和非成员。

成员推理攻击在开山之作中被这样解释：定义攻击模型f,attack()，它的输入X,attack是一个由正确的标签类和一个目标模型(被攻击模型)的预测置信度向量组成(后面我们会看到，其实不一定非得如此，这也是成员推理攻击的tricks之一),该攻击模型的输出为一个预测类"in"(member)或"out"(non-member)。

### 核心思想

成员推断攻击模型的训练，既需要样本真实label，又需要目标模型预测置信度向量，对于攻击者的要求是比较苛刻的，在现实中是很难实现的。为了解决这些问题，Shokri等人十分巧妙的提出了一个核心思想——shadow model，这是使得开山之作中成员推理攻击能够work的关键。

所谓shadow model，即我们可以不适用目标模型的训练数据以及具体的模型参数(后续有研究证明，甚至可以不使用与目标模型相同的模型，比如目标模型实际为ResNet，这在现实中我们也是很难得到的，我们可以使用DenseNet或是简单的CNN，也能达到较好的攻击效果)。我们可以使用自己的数据样本(格式要与目标模型的训练样本相同，这是比较容易得到的)训练与目标模型架构相同的模型，然后使得shadow model对于数据样本的输出与目标模型相近即可。作者认为，现在机器学习模型的学习能力已经足够的强，相同模型使用相同分布的数据样本进行训练，若模型对于数据的表现(攻击者通常可以使用目标模型进行黑盒查询)也相近，那么模型其实也是近乎等价的。因此我们可以使用shadow training dataset来近似替代目标模型训练集，使用shadow model的预测置信度来代替目标模型的预测置信度，以此来训练我们的攻击模型。文章也进行了大量的实验，实验结果表明使用shadow model确实有不错的效果。但是其实还有一个问题，就是shadow training dataset也是比较难获得的。

### 分类

如果以攻击者是否知道目标模型的具体权值参数作为区分标准，我们可以将成员推理攻击归为两种：黑盒攻击和白盒攻击(还有一种灰盒攻击，比较少见，这里不做阐述)。所谓黑盒攻击就是攻击者对于目标模型的具体权值参数无法获知，只能向目标模型查询来获得目标模型对于数据样本的预测结果。

攻击模型是一组模型，为目标模型的每个输出类设置一个。这增加了攻击的准确性，因为目标模型根据输入的真实的类从而在其输出的类上产生不同的分布。

为了训练的攻击模型，本文将介绍多个影子模型(shadow model)，这些模型的行为类似于目标模型，与目标模型相比每个影子模型的真实情况是已知的，即给定的记录是否在其训练数据集中。因此，可以对影子模型的输入和相应的输出(每个标记为“in”或“out”)进行监督训练，教攻击模型如何区分影子模型对其训练数据集成员的输出和对非成员的输出，如下图：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo28.png" width="50%">

攻击者使用数据记录查询目标模型，并获取该记录的模型预测。预测是一个概率向量，每类一个，表明记录属于某个类。该预测向量连同目标记录的标签被传递到攻击模型，该攻击模型推断该记录是否在目标模型的训练数据集中。

## 影子模型

### 影子模型概述

影子模型必须以与目标模型类似的方式进行训练。如果目标的训练算法(例如：神经网络、SVM、逻辑回归)和模型结构(例如神经网络的连线)是已知的，那这将很容易。机器学习即服务则更具挑战性，因为这里不知道目标模型的类型和结构，但攻击者可以使用完全相同的服务，像用于训练目标模型那样训练影子模型，见下图：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo29.png" width="50%">

目标模型和影子模型的训练数据集具有相同的格式但不相交。阴影模型的训练数据集可以重叠。所有模型的内部参数都是独立训练的。

影子模型越多，攻击模型就越精确。对攻击模型进行了训练影子模型的行为上的差异，更多的影子模型为攻击模型提供了更多的训练素材。

### 生成影子模型的训练数据

为了训练影子模型，攻击者需要与目标模型的训练数据相似地分布的训练数据。如下有几种生成此类数据的方法：

**A、基于模型的合成(Model-based synthesis)**

如果攻击者没有真正的训练数据，也没有关于其分布的任何统计，他可以利用目标模型本身为影子模型生成合成训练数据。直观来看，高置信度(high confidence)的目标模型分类的记录应当在统计上类似于目标的训练数据集，从而为影子模型提供良好的素材。

合成过程分为两个阶段：(1)利用爬山算法(hill-climbing)对可能的数据记录空间进行搜索，找出具有较高置信度的目标模型分类输入；(2)从这些记录中抽取合成数据。在此过程合成记录之后，攻击者可以重复该记录，直到影子模型的训练数据集已满为止。

```
注意：只有当敌手能够有效地探索可能输入的空间并发现被目标模型分类的高可信度的输入时，该合成过程才起作用。例如，如果输入是高分辨率图像，并且目标模型执行复杂的图像分类任务，则可能无法起作用。
```

**B、基于统计的合成(Statistics-based synthesis)**

攻击者可能拥有被提取目标模型所需训练数据的总体集(population)的一些统计信息。例如，攻击者可以事先知道不同特征的边缘分布。在实验中，通过独立地从每个特征的边缘分布中采样的值来生成影子模型的合成训练记录。由此产生的攻击模型很有效。

**C、有噪的真实数据(Noisy real data)**

攻击者可以访问一些与目标模型的训练数据类似的数据，这些数据可以被视为“噪音”版本。此场景模拟的情况是，目标模型和影子模型的训练数据没有从完全相同的集合中采样，或者以非均匀的方式进行采样。

### 训练攻击模型

影子训练技术的中心思想是使用相同服务在相对类似的数据记录上训练的类似模型以类似的方式表现出来。结果表明，学习如何推理影子模型训练数据集中的成员会得到一种攻击模型，该模型也成功地推断出目标模型训练数据集中的成员。

使用自己的训练数据集和相同大小的不相交测试集来查询每个影子模型。训练数据集上的输出被标记为in，其余的标记为out。现在，攻击者拥有记录的数据集、影子模型的相应输出和输入/输出标签。攻击模型的目标是从记录和相应的输出中推断出标签，如下图：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo30.png" width="50%">

实验中将识别训练数据集成员与模型输出之间复杂关系转化为一个二分类问题。二分类是一项标准的机器学习任务，因此，可以使用任何先进的机器学习框架或服务来建立攻击模型，且该方法不依赖于攻击模型训练的具体方法。

### 评价

使用标准的精确率(precision)和召回度(recall)来评估攻击。精确率是作为训练数据集中确实是成员的成员推理得到的记录的一部分(作为成员推断的记录中，有多少部分确实是训练数据集的成员)；而召回度度量的是攻击覆盖率，即攻击者可以正确推断为成员的训练记录的一部分(攻击者正确推断训练数据集的哪些部分为成员)。大多数度量都是按类报告的，因为攻击的准确性可能因不同类别而有很大差异。这是由于属于每个类的训练数据的大小和组成不同，并且高度依赖于数据集。

对于大多数目标模型的类，推理攻击达到了很高的精确率。这表明，如果攻击者能够高效地生成被目标模型分类的高置信度的输入，成员推理攻击可以仅通过黑盒访问目标模型来进行训练，而不需要事先了解目标模型的训练数据的分布情况。

**(1)影子训练数据的影响**

在训练数据集是真实数据的噪声版本的影子模型上训练的攻击中，随着噪声量的增加，精度下降，但是与原攻击相比较，影子的训练数据中有10%的特征被随机值所取代的情况下攻击始终高于基准。这表明即使攻击者对目标模型的训练数据分布的假设不太准确，攻击也是健壮的。

**(2)每类中训练数据数量和分类数量的影响**

目标模型的输出的分类的数量有助于了解模型泄漏的程度。越多分类，攻击者就能获得更多关于模型内部状态的信号。每个类训练数据的数量与成员推理准确度之间的关系比较复杂，但一般来说，训练数据集中与给定类关联的数据越多，该类的攻击精度就越低。

**(3)过拟合的影响**

模型越过拟合，泄漏问题越严重，但只适用于同一类型的模型，过度拟合不是导致模型容易受到成员推理的唯一因素，还有模型的结构和类型也造成了这个问题。

本文介绍了针对机器学习模型的成员关系推理攻击，此类攻击是一种通用的、定量的方法，用于了解机器学习模型如何泄漏关于其训练数据集的信息。当选择要训练的模型类型或要使用的机器学习服务时，这种攻击可以作为选择指标之一。关键技术创新是影子训练技术，该技术训练攻击模型，以区分目标模型在训练数据集的成员和非成员上的输出。在这种攻击中使用的影子模型可以有效地利用合成或噪声数据创建，对于从目标模型本身生成的合成数据，攻击不需要事先知道目标模型的训练数据的分布情况。从隐私角度来看一些数据集的成员是非常敏感的（如医患），因此研究结果具有很强的现实意义。

## 成员推断攻击实现

### 总体概述

本项目旨在实现模型的成员推断攻击，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、FashionMNIST等数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN等数据集。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/MIA_model](https://xihe.mindspore.cn/projects/hepucuncao/MIA_model)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记README文档，以及ResNet模型的模型训练和推理代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── classifier_method.py    # 模型训练和评估框架
 │  └── cnn_model.py    # cnn网络模型代码
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

2.同时可以运行cnn_model.py程序以初始化CNN卷积神经网络模型的参数，该程序创建了一个简单的CNN模型，用于执行图像分类任务。模型首先对输入的图像数据进行卷积和池化操作，然后将数据输入到全连接层中以进行进一步的处理，最终输出概率分布。

初始化网络结构：
- self.conv1：一个卷积层(Conv2d)，接受3通道(RGB)的图像输入，输出6个通道，使用5x5的滤波器(kernel)。
- self.pool：一个最大池化层(MaxPool2d)，每次下采样2x2。
- self.conv2：第二个卷积层，输出16个通道，同样使用5x5的滤波器。
- self.fc1：一个全连接层(Linear)，将卷积层输出展平后连接到120个神经元。
- self.fc2：第二个全连接层，连接到84个神经元。
- self.fc3：第三个全连接层，连接到10个神经元，通常用于分类任务，每个输出对应一个类别。

forward方法定义了网络的前向传播过程：

首先，对输入x应用ReLU激活函数，然后通过conv1层；应用最大池化层，进一步下采样；再次通过ReLU激活函数和conv2层；将卷积层的输出展平(view操作)，将2D特征图展平成1D向量，以便全连接层处理(这里展平后的维度为5x5或4x4乘以16通道,取决于conv1的输出);应用ReLU激活函数到fc1层；再次通过ReLU激活函数到fc2层；最后，通过fc3层并返回输出。

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

在classifier_methods.py程序中，定义了训练过程，接受多个参数，如模型类型('fc' 或 'cnn')、隐藏层维度(fc_dim_hidden)、输入和输出维度(fc_dim_in 和 fc_dim_out)、批大小(batch_size)、训练轮数(epochs)、学习率(learning_rate)等。根据模型类型创建网络(FCNet/CNN）)将网络移到可用的GPU/CPU。然后对训练数据和测试数据进行迭代，计算损失并更新模型参数。在训练结束时，计算并打印训练集和测试集的准确率。

### 结果分析

本项目将以经典的多通道数据集CIFAR-10数据集为例，展示代码的执行过程并分析其输出结果。

首先要进行run_attack.py程序中一些参数和路径的定义，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo17.png" width="50%">

全部程序运行完毕后，可以看到控制台打印出的信息，下面具体分析输出的结果。

首先是一组参数（字典）的输出，这些参数定义了模型训练的配置：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo8.png" width="50%">

其中target_model: 目标模型(例如CNN);target_learning_rate: 目标模型的学习率;target_epochs: 目标模型训练的轮数;n_shadow: 阴影模型的数量;attack_model: 攻击模型(例如FC，全连接模型);attack_epochs: 攻击模型训练的轮数，等等。

接着开始训练目标模型，输出显示了目标模型在训练集和测试集上的准确率：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo9.png" width="50%">

开始训练阴影模型，每训练一个阴影模型(如0到9)，都会输出类似的信息，展示了该阴影模型在训练集和测试集上的准确率，并表明训练完成。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo10.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo11.png" width="50%">

训练所有阴影模型后，继续训练攻击模型，训练了针对每个类别的攻击模型，并输出每个类别的训练集和测试集准确率。同时，还会输出用于训练和测试的数据集中的样本数量，这些数字对于评估模型的性能非常重要。通常，训练集用于调整模型参数，而测试集用于评估模型在未见过的数据上的泛化能力。在理想情况下，测试集应该足够大，以便能够提供对模型性能的可靠估计，训练集也应该足够大，以便模型能够学习到数据中的模式和特征。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo12.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo13.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo14.png" width="50%">
<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo15.png" width="50%">

最后打印出分类报告：输出了精确度、召回率、F1分数、支持度等指标，整体准确率在0.60附近。整体来看，模型的表现还有提升的空间，可以进一步优化模型参数和训练策略。

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo16.png" width="50%">

## 复杂场景下的成员推断攻击

### 总体介绍

该过程主要是在LeNet5模型和ResNet模型的基础之上开启复杂场景下的成员推断攻击，以经典数据集MNIST为例。

首先，分别对LeNet5模型和ResNet模型的训练数据集随机删除5%和10%的数据，记录删除了哪些数据，并分别用剩余数据重新训练LeNet5和ResNet模型，形成的模型包括原LeNet5模型，删除5%数据后训练的LeNet5模型，删除10%数据后训练的LeNet5模型，原ResNet模型，删除5%数据后训练的ResNet模型，删除10%数据后训练的ResNet模型。然后，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和之后训练而成的模型的攻击成功率。最后，记录攻击对比情况。

上述是完全重训练的方式，即自动化地实现删除，并用剩余数据集训练一个新模型，并保存新模型和原来的模型。本文还采用了其他更快的方法，即分组重训练，具体思路为将数据分组后，假定设置被删除数据落在某一个特定组，重新训练时只需针对该组进行重训练，而不用完全重训练。同样地，保存原来模型和分组重训练后的模型。

### 代码结构
```python
 ├── MIA    # 相关代码目录
 │  ├── LeNet5   # LeNet5模型训练代码
 │      └── net.py    # LeNet5网络模型代码
 │      └── lenet_train.py     # LeNet5模型完全重训练代码
 │      └── lenet_part_train.py   #LeNet5模型分组重训练代码
 ├  ├── ResNet   # ResNet模型训练代码
 │      └── resnet_train.py     # ResNet模型完全重训练代码
 │      └── resnet_part_train.py   #ResNet模型分组重训练代码
 ├  ├── MIA_attack  # 复杂场景下的成员推理攻击代码
 │      └── cnn_model.py    # 各种网络模型代码
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

2.然后通过改变percentage的值，生成对未删除数据的数据集、随机删除5%数据后的数据集和随机删除10%数据后的数据集，然后重新训练LeNet5和ResNet模型，形成的模型包括原LeNet5模型，删除5%数据后训练的LeNet5模型，删除10%数据后训练的LeNet5模型，原ResNet模型，删除5%数据后训练的ResNet模型，删除10%数据后训练的ResNet模型。

具体训练步骤与原来无异，区别在于要调用remove_data函数生成删除数据后的数据集，举例如下：
```

remaining_train_dataset_5, removed_indices_5 = remove_data(train_dataset, 0.05)
train_dataloader_5 = torch.utils.data.DataLoader(remaining_train_dataset_5, batch_size=16, shuffle=True)

注意：如果是在同一个程序中生成用不同数据集训练的模型，要记得在前一个模型训练完之后重新初始化模型，如model = MyLeNet5().to(device)，且删除5%和10%数据都是在原数据集的基础上，而不是叠加删除。

```

3.利用前面讲到的模型成员攻击算法，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和删除之后训练而成的模型的攻击成功率，并记录攻击的对比情况。

具体攻击的方法和步骤和前面讲的差不多，不同点在于，由于这里我们用的训练模型是LeNet5模型以及ResNet模型，所以我们在cnn_model.py中要构造这两种模型的网络模型。相应地，对不同模型进行成员推理攻击时在classifier_methods.py程序中初始化的网络模型也不同，如攻击ResNet模型时网络初始化代码为net = ResNet18()。

### 结果记录及分析

1.首先比较删除数据前后LeNet5模型的训练准确率，如下图所示：

(1)完全重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/LeNet5/photo14.png" width="30%">

(图1：未删除数据的LeNet5模型训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo18.jpg" width="30%">

(图2：删除5%数据后的LeNet5训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo19.jpg" width="30%">

(图3：删除10%数据后的LeNet5训练准确率)

由上述结果可以看出，随着删除数据的比例增加，模型训练的精确度是随之降低的。

为了验证上面的结果是否具有普遍性，我们应该尝试更多种类数据集和更多种类的模型，这里笔者又使用多通道数据集CIFAR-10进行数据的删除及模型的训练，结果如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo35.png" width="30%">

(图4：未删除数据的LeNet5模型训练准确率-cifar10)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo36.jpg" width="30%">

(图5：删除5%数据后的LeNet5训练准确率-cifar10)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo37.png" width="30%">

(图6：删除10%数据后的LeNet5训练准确率-cifar10)

由上述结果可以看出，删除数据后模型训练的精确度反而升高了，且随着删除数据的比例增加，模型训练的精确度也是随之升高的。这也说明了数据的数量和模型训练精度的关系不是线性的，它们之间存在复杂的关系，需要更多的尝试来探寻它们之间的联系，而不能一概而论！

(2)分组重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo31.png" width="30%">

(图7：删除5%数据后的LeNet5训练准确率，这里随机删除了第2组的数据)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo32.png" width="30%">

(图8：删除10%数据后的LeNet5训练准确率，这里随机删除了9组的数据)

训练过程中我们可以明显感觉到，采用分组重训练的方式，模型训练的速度比完全重训练快得多！

2.接着比较删除数据前后ResNet模型的训练准确率，如下图所示：

(1)完全重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo20.jpg" width="30%">

(图9：未删除数据的ResNet模型训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo22.jpg" width="30%">

(图10：删除5%数据后的ResNet训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo21.png" width="30%">

(图11：删除10%数据后的ResNet训练准确率)

同样地，为进一步说明结论，下面是使用KMNIST数据集得出的结果：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo38.png" width="30%">

(图12：未删除数据的ResNet模型训练准确率-KMNIST)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo39.png" width="30%">

(图13：删除5%数据后的ResNet训练准确率-KMNIST)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo40.png" width="30%">

(图14：删除5%数据后的ResNet训练准确率-KMNIST)

由上述结果可知，随着删除数据的比例增加，模型的精确度可能会升高，也可能会降低，这取决于删除的数据对模型的影响。

(2)分组重训练

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo33.png" width="30%">

(图15：删除5%数据后的ResNet训练准确率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo34.png" width="30%">

(图16：删除10%数据后的ResNet训练准确率)

```
如果删除的数据是噪音数据或outliers，即不具代表性的数据，那么删除这些数据可能会提高模型的精确度。因为这些数据可能会干扰模型的训练，使模型学习到不正确的规律。删除这些数据后，模型可以更好地学习到数据的模式，从而提高精确度。

但是，如果删除的数据是重要的或具代表性的数据，那么删除这些数据可能会降低模型的精确度。因为这些数据可能包含重要的信息，如果删除这些数据，模型可能无法学习到这些信息，从而降低精确度。

此外，删除数据还可能会导致模型的过拟合，即模型过于拟合训练数据，无法泛化到新的数据上。这是因为删除数据后，模型可能会过于依赖剩余的数据，导致模型的泛化能力下降。
```

3.然后开始对形成的模型进行成员推理攻击，首先比较删除数据前后训练而成的LeNet5模型的攻击成功率，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo23.jpg" width="30%">

(图17：未删除数据的LeNet5模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo24.jpg" width="30%">

(图18：删除5%数据后的LeNet5模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo25.jpg" width="30%">

(图19：删除10%数据后的LeNet5模型攻击成功率)

由上述结果可知，随着删除数据的比例增加，模型成员推断攻击的成功率有细微地降低，但删除5%数据后的LeNet5模型攻击成功率比删除10%后的低。

4.接着比较删除数据前后训练而成的ResNet模型的攻击成功率，如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo27.jpg" width="30%">

(图20：未删除数据的ResNet模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo28.jpg" width="30%">

(图21：删除5%数据后的ResNet模型攻击成功率)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo29.jpg" width="30%">

(图22：删除10%数据后的ResNet模型攻击成功率)

由上述结果可知，随着删除数据的比例增加，模型成员推断攻击的成功率有细微地降低，但删除5%数据后的LeNet5模型攻击成功率和删除10%后的几乎一样。

为了更好地得出结论，我们还可以多尝试更多种的数据集类型和模型，这里笔者又用多通道数据集CIFAR-10进行了验证，得到的结果如下图所示：

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo23.jpg" width="30%">

(图17：未删除数据的LeNet5模型攻击成功率-cifar10)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo24.jpg" width="30%">

(图18：删除5%数据后的LeNet5模型攻击成功率-cifar10)

<img src="https://hepucuncao.obs.cn-south-1.myhuaweicloud.com/MIA/photo25.jpg" width="30%">

(图19：删除10%数据后的LeNet5模型攻击成功率-cifar10)

```
删除一部分数据再进行模型成员推断攻击，攻击的成功率可能会降低。这是因为模型成员推断攻击的原理是利用模型对训练数据的记忆，通过观察模型对输入数据的行为来判断该数据是否在模型的训练集中。

如果删除了一部分数据，模型的训练集就会减少，模型对剩余数据的记忆就会减弱。这样，攻击者就更难以通过观察模型的行为来判断某个数据是否在模型的训练集中，从而降低攻击的成功率。

此外，删除数据还可能会使模型变得更robust，对抗攻击的能力更强。因为模型在训练时需要适应新的数据分布，模型的泛化能力就会提高，从而使攻击者更难以成功地进行成员推断攻击。

但是，需要注意的是，如果删除的数据是攻击者已经知晓的数据，那么攻击的成功率可能不会降低。因为攻击者已经知道这些数据的信息，仍然可以使用这些信息来进行攻击。

本项目所采用的模型都是神经网络类的，如果采用非神经网络类的模型，例如，决策树、K- means等，可能会有不一样的攻击效果，读者可以尝试一下更多类型的模型观察一下。
```


