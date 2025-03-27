# Transformer_AI

2024年11月15日**更新**

在此教程中，我们将对基于Transformer的生成式人工智能模型及其原理进行一个简单的介绍，并实现其训练和推理过程，且至少支持3种数据集，目前支持数据集有：MNIST、fashionMNIST、CIFAR10等，并给用户提供一个详细的帮助文档。

## 目录  

[基本介绍](#基本介绍)  
- [什么是生成式AI](#什么是生成式AI)
- [生成式AI的类型及应用场景](#生成式AI的类型及应用场景)
- [什么是Transformer架构](#什么是Transformer架构)
- [基于Transformer的生成式AI](#基于Transformer的生成式AI)

[Transformer_AI实现](#Transformer_AI实现)
- [总体概述](#总体概述)
- [项目地址](#项目地址)
- [项目结构](#项目结构)
- [训练及推理步骤](#训练及推理步骤)
- [实例](#实例)

[成员推断攻击实现](#成员推断攻击实现)
- [总体介绍](#总体介绍)
- [MIA项目结构](#MIA项目结构)
- [实现步骤及分析](#实现步骤及分析)
- [结果分析](#结果分析)

[复杂场景下的成员推断攻击](#复杂场景下的成员推断攻击)
- [介绍](#介绍)
- [代码结构](#代码结构)
- [实现步骤](#实现步骤)
- [结果记录及分析](#结果记录及分析)

[文本处理](#文本处理)
## 基本介绍

### 什么是生成式AI

生成式AI是人工智能的一个分支，可以根据已经学习的内容生成新的内容。它从现有的内容中学习的过程叫做训练，训练的结果是创建一个统计模型。当用户给出提示词时，生成式AI将会使用统计模型去预测答案，生成新的文本来回答问题。

生成式AI主要可以分成两类：

**生成式语言模型**：这是基于自然语言处理的技术，通过学习语言的规律和模式来生成新的文本，它可以根据之前的上下文和语义理解生成连贯的句子或段落。生成式语言模型的训练基于大规模的文本数据，例如新闻文章、小说或网页内容。通过学习文本中的单词、短语和句子之间的关系，生成式语言模型可以自动生成新的、具有逻辑和语法正确性的文本，如文章、对话和诗歌等。

**生成式图片模型**：这是基于计算机视觉的技术，通过学习图像的特征和结构来生成新的图像。它可以从之前的训练数据中学习到图像的特征表示和统计规律，然后使用这些知识生成新的图像。生成式图片模型的训练通常基于大规模的图像数据集，例如自然图像或艺术作品。通过学习图像的纹理、颜色、形状和物体之间的关系，生成式图片模型可以生成具有视觉真实感或艺术风格的新图像，如自然风景、人像或抽象艺术作品等。

其工作原理包括数据收集与预处理、模型选择(如生成对抗网络、变分自编码器和自回归模型)、模型训练(通过对抗过程或编码-解码机制逐步提高生成质量)，以及生成数据的过程(通过随机采样和温度调节控制多样性)。训练完成后，模型能够生成逼真的文本、图像或音频等数据，广泛应用于自动写作、图像生成和音乐创作等领域。

### 生成式AI的类型及应用场景

1.文本到文本生成模型旨在接收一个文本输入，并生成一个相关的文本输出。这种模型可用于机器翻译、文本摘要、对话生成、故事生成等任务。生成模型可以学习从输入到输出的映射关系，以生成具有语义和语法正确性的新文本。

常见应用场景：

- 机器翻译：将一种语言的文本翻译成另一种语言。
- 文本摘要：从长篇文本中生成简洁的摘要或概括。
- 对话生成：生成自然流畅的对话，可用于虚拟助手或聊天机器人。
- 故事生成：自动生成连贯、有趣的故事或叙述。

2.文本到视频或三维生成模型接收一个文本输入，并生成相应的视频或三维模型输出。这些模型可以用于视频生成、场景合成、三维模型生成等任务。模型可以学习从文本描述到视频序列或三维模型的转换过程，生成与文本描述相符的动态视频或立体模型。

常见应用场景：

- 视频生成：根据文本描述生成与之相符的动态视频。
- 场景合成：根据文本描述生成三维场景或虚拟现实体验。
- 三维模型生成：根据文本描述生成具有特定属性或形状的三维模型。

3.文本到任务生成模型旨在根据文本输入执行特定任务。这些模型可以接收自然语言指令或问题，并生成相应的任务执行结果。例如，问答生成模型可以接收问题，并生成相应的答案；代码生成模型可以接收自然语言描述，并生成相应的代码实现。这种模型能够将文本指令转化为任务执行的具体操作。

常见应用场景：

- 问答生成：根据问题生成相应的答案或解决方案。
- 代码生成：将自然语言描述转化为代码实现。
- 指令执行：根据自然语言指令执行特定的任务，如图像处理、数据操作等。

### 什么是Transformer架构

Transformer模型本质上都是预训练语言模型，大都采用自监督学习(Self-supervised learning)的方式在大量生语料上进行训练，也就是说，训练这些Transformer模型完全不需要人工标注数据。Transformer架构分为两个主要部分：编码器(Encoder)和解码器(Decoder)，其架构的主要组成部分如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo1.png" width="40%">

- Encoder(左边)：负责理解输入文本，为每个输入构造对应的语义表示(语义特征)；
- Decoder(右边)：负责生成输出，使用Encoder输出的语义表示结合其他输入来生成目标序列。

Transformer架构的核心优势在于其能够学习句子中所有单词的相关性和上下文，而不仅仅是相邻单词之间的关系。通过应用注意力权重，模型可以学习输入中每个单词与其他单词的相关性，无论它们在输入中的位置如何。这使得模型能够理解句子的整体意义和上下文，从而提高了语言编码的能力。

相比之前占领市场的LSTM和GRU模型，Transformer有两个显著的优势:

- Transformer能够利用分布式GPU进行并行训练，提升模型训练效率；
- 在分析预测更长的文本时, 捕捉间隔较长的语义关联效果更好。

### 基于Transformer的生成式AI

让生成式人工智能展现出流利的语言理解和生成能力的关键就是“Transformer”这一工具，它极大地加速和增强了计算机处理语言的方式。“Transformer” 架构的关键在于一种叫做 “自注意力(Self-attention)” 的概念，它使生成式能理解单词之间的关系。Transformer可以一次性地处理一个完整的序列，比如一句话、一个段落或是一篇完整的文章，并分析其中的每个部分而不只是单个词语。这样，软件就能更好地把握上下文和模式，从而更准确地翻译或生成文本。这种一次性处理的方式也使得LLMs训练得更快，进而提高了它们的效率和可扩展性。

在这过程中，“自注意力”机制会分析文本的每一个 token，并判断哪些部分对理解其整体含义更为重要。这种功能对于创建高质量的文本非常关键。没有这个功能，某些词可能会在不适合的情境下被曲解。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo2.png" width="40%">

在 “Transformer” 出现之前，先进的AI翻译主要依赖于循环神经网络(RNNs)，它们会逐词处理句子。然而，“Transformer” 通过“自注意力”机制，可以同时处理句子中的所有词，捕捉到更多的上下文信息，从而让生成式AI拥有更强大的语言处理能力。

例如：生成式预训练Transformer模型，通常称为GPT，是一系列使用Transformer架构的神经网络模型，是为ChatGPT等生成式人工智能应用程序提供支持的人工智能的一项关键进展。GPT模型使应用程序能够创建类似人类的文本和内容(图像、音乐等)，并以对话方式回答问题。

## Transformer_AI实现

### 总体概述

本项目旨在实现基于Transformer的生成式人工智能模型，并且支持多种数据集，目前该模型可以支持各种图像分类数据集，如单通道的数据集MNISTT等，以及多通道的数据集CIFAR10、CIFAR100等。模型最终将数据集分类为100种类别，可以根据需要增加分类数量。训练轮次默认为15轮，同样可以根据需要增加训练轮次。

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/hepucuncao/DeepLearning](https://xihe.mindspore.cn/projects/hepucuncao/Transformer_AI)

<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：学习笔记README文档，以及基于Transformer的AI模型的模型训练和推理代码放在train文件夹下。

```python
 ├── train    # 相关代码目录
 │  ├── net.py    # Transformer网络模型代码
 │  ├── train.py    # 模型训练代码
 │  └── test.py    # 模型推理代码
 │  └── utils.py    # 模型评估代码
 │  └── dataset.py    # 数据处理代码
 │  └── text.py    # 文本数据训练代码
 └── README.md 
```

### 训练及推理步骤

- 1.首先运行net.py初始化Transformer网络模型的各参数；

这里要注意的是，需要提前下载预训练权重，在net.py文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重。

- 2.接着运行utils.py文件，主要用于处理图像数据集，执行训练和评估模型的操作。

这里函数要读取class_indices.json文件，获取类别索引与类别名的映射关系，因此要提前设置好json文件的路径，即json_path = '文件路径'。

- 3.接着运行train.py会进行模型训练，要加载的训练数据集和测试训练集可以自己选择，本项目可以使用的数据集来源于torchvision的datasets库。相关代码如下：

```

训练数据集：train_dataset = datasets.数据集名称(root='下载路径', train=True, download=True, transform=data_transform["train"])
推理数据集：val_dataset = datasets.数据集名称(root='下载路径', train=False, download=True, transform=data_transform["val"])

这里理论上可以尝试任意多种数据集，只需把数据集名称更换成你要使用的数据集(datasets中的数据集)，并修改下载数据集的位置(默认在根目录下，如果路径不存在会自动创建)即可，如果已经提前下载好了则不会下载，否则会自动下载数据集。

```

另外，还要创建模型保存目录，即if not os.path.exists("保存路径"): os.makedirs("保存路径")，可以自定义文件路径，该代码保证了如果路径不存在的话会自动创建。程序在每个训练周期后会保存模型的权重，因此要设置好模型保存的路径，即torch.save(model.state_dict(), "文件保存路径+文件名称".format(epoch))。

在主函数中还需要定义训练参数，如下：
    class Args:
        num_classes = 分类类别数
        epochs = 训练轮数
        batch_size = 每次训练过程中用于计算梯度的样本数量
        lr = 学习率
        lrf = 学习率的最终值
        model_name = "模型名称(可省)"
        weights = '预训练权重文件路径'
        freeze_layers = 指示是否在训练过程中冻结某些模型层的参数(如果设置为True，则在训练时不会更新这些层的权重，常用于迁移学习以防止过拟合)
        device = "指定计算设备"


同时，程序会实时将每一轮训练的损失值和精确度打印出来，训练完成后显示最终值，损失值越接近0，则说明训练越成功，从进度条也可以看出本轮的训练进度。

- 4.由于train.py代码会将精确度最高的模型权重参数保存下来，以便推理的时候直接使用模型，因此运行train.py之前，需要设置好保存的路径，相关代码如下：

```

torch.save(model.state_dict(),'保存路径及名称')

默认保存路径为根目录，可以根据需要自己修改路径，该文件夹不存在时程序会自动创建。

```

- 5.保存完毕后，我们可以运行test.py代码进行模型的推理，同样需要加载数据集(和训练过程的数据相同)，步骤同2。同时，我们应将保存的最好模型权重文件加载进来，相关代码如下：

```

model_weight_path = "文件路径"
assert os.path.exists(model_weight_path), f"Model weights '{model_weight_path}' not found."
model.load_state_dict(torch.load(model_weight_path, map_location=device))

文件路径为最好权重模型的路径，注意这里要写绝对路径，并且windows系统要求路径中的斜杠应为反斜杠。

```

另外，程序还设置了函数plot_metrics_per_class绘制每个类别的精确度、召回率和F1分数的柱状图，以及plot_metrics绘制加权平均的精确度、召回率和F1分数的柱状图，同时计算并打印准确率。

## 实例

这里我们以一个经典图像数据集CIFAR100为例：

首先，我们要提前下载预训练权重文件，这里我们选择的是vit_base_patch16_224_in21k，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo3.png" width="40%">

```

def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model

```

运行train.py之前，要加载好要训练的数据集，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo4.png" width="50%">

设置训练好的最好模型权重参数pth文件的保存路径以及主函数中的训练参数：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo5.png" width="50%">

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo6.png" width="50%">

这里我们设置训练轮次为15，如果没有提前下载好数据集，程序会自动下载在我们设置好的目录下。

训练好的模型权重参数保存在设置好的路径中：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo7.png" width="50%">

正式运行的时候，我们要传入参数：将--data-path设置成解压后的训练集文件夹绝对路径、--weights参数设成下载好的预训练权重路径，设置好就能使用train.py脚本开始训练了。

从下图最后一轮的损失值和精确度可以看出，精确度大多都可以保持在80%附近，可见训练的结果是较为准确的。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo8.png" width="50%">

同时，训练过程中会自动生成class_indices.json文件，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo12.png" width="50%">


最后我们运行test.py程序，首先要把train.py运行后保存好的权重参数文件加载进来(默认保存在weights文件夹下)，设置的参数如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo9.png" width="50%">

设置好之后运行即可对测试集准确率、召回率、F1分数进行计算（包括对每个类别的三种指标计算以及平均结果），并输出其柱形图，同时输出混淆矩阵。

由下图最终的运行结果我们可以看出，测试的结果是较为准确的。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo10.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo11.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo13.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo14.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo15.png" width="50%">

其他数据集的训练和推理步骤和CIFAR100数据集大同小异。

## 成员推断攻击实现

## 成员推断攻击实现

### 总体介绍

本项目旨在实现强化学习模型的成员推断攻击，并且支持多种数据集，目前该模型可以支持单通道的数据集，如：MNIST、FashionMNIST等数据集，也可以支持多通道的数据集，如：CIFAR10、SVHN等数据集。同时，本项目还就如何提高攻击准确率进行讨论。

<a name="MIA项目结构"></a>
### MIA项目结构

项目的目录分为两个部分：学习笔记README文档，以及RL模型的模型训练和推理代码放在MIA文件夹下。

```python
 ├── MIA    # 相关代码目录
 │  └── net.py    # 网络模型代码
 │  └── run_attack.py   #成员推断攻击代码
 └── README.md 
```

### 实现步骤及分析

1.首先运行net.py程序，该程序实现了一个基于Vision Transformer(ViT) 的模型，主要用于图像分类任务，并且通过各种模块(如注意力、MLP、路径丢弃等)来增强模型的表现。每个模块的设计都遵循了 Transformer 的架构理念，并且代码中提供了多个函数来实例化不同配置的模型。

主要模块有以下几个：

- drop_path函数：实现了路径丢弃(也称为随机深度)，用于在训练时以一定的概率随机丢弃路径，这种方法有助于防止过拟合。
- PatchEmbed类：负责将输入图像切分成小块(patches)，并将每个小块嵌入到一个向量空间中。该类使用卷积层来实现这一功能。
- Attention类：实现了多头自注意力机制。输入的特征通过线性变换生成查询(Q)、键(K)和值(V)，并计算注意力权重。最后，通过对值的加权求和获得输出。
- 前馈神经网络(MLP)：实现了一个简单的前馈神经网络，包括两个线性层和一个激活函数（默认为 GELU），用于在 Transformer 的每个块中进行非线性变换。
- Transformer块：结合了注意力机制和前馈神经网络，并且在每个子层后添加了残差连接。它还支持路径丢弃。
- Vision Transformer模型：是整个模型的核心，负责定义模型的架构，包括嵌入层、多个 Transformer 块、分类头和位置嵌入。这个类也处理模型的输入和输出。
- 模型实例化函数(如 vit_base_patch16_224 等)：这些函数用于创建不同版本的 ViT 模型，例如基础模型和大型模型，允许指定不同的类数和其他参数。

2.接着运行run.attack.py程序，代码主要实现了一个攻击模型的训练过程，包括目标模型、阴影模型和攻击模型的训练，可以根据给定的参数设置进行模型训练和评估。

运行代码之前，要先定义一些常量和路径，包括训练和测试集的大小、模型保存路径、类别数、批次大小以及设备等，数据集若未提前下载程序会自动下载，相关代码如下：

```
# Set seeds for reproducibility
torch.manual_seed(171717)
np.random.seed(171717)

# CONSTANTS
TRAIN_SIZE = 训练集大小
TEST_SIZE = 测试集大小
MODEL_PATH = '模型保存路径'

num_classes = 类别数
batch_size = 批次大小
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载数据集
trainset = torchvision.datasets.数据集名称(root='保存路径', train=True, download=True,
                                            transform=transform)

testset = torchvision.datasets.数据集名称(root='保存路径', train=False, download=True,
                                           transform=transform)

np.savez(MODEL_PATH + '保存文件名称', attack_x=attack_x, attack_y=attack_y, classes=classes)

np.savez(MODEL_PATH + '保存文件名称', shadow_attack_x=shadow_attack_x, shadow_attack_y=shadow_attack_y,
             shadow_classes=shadow_classes)

```

其中，程序定义了generate_data_indices函数，以随机选择目标模型的训练数据和影子模型的数据。train_vit_model函数用于训练给定的ViT模型，计算训练损失，并在验证集上评估模型的准确性。准备攻击数据用prepare_attack_data函数通过模型的输出(logits)准备攻击数据，标记数据为目标类或非目标类(1和0)。train_target_model函数训练目标模型并准备攻击数据，保存攻击数据到指定路径。train_shadow_models函数训练多个影子模型，并准备影子模型的攻击数据。train_attack_model函数使用Logistic Regression作为攻击模型，并标准化攻击数据，训练后评估攻击模型的性能

```

为了提高攻击的精确度，本项目的代码在之前的成员推理攻击的代码上做了修改，包括以下几方面：

1.准备攻击数据：在prepare_attack_data函数中，通过提取模型输出的logits作为攻击模型的输入特征。这些特征包含了模型对样本的信心度，有助于攻击模型判断样本是否属于目标类。

2.影子模型的多样性：训练多个影子模型（在代码中默认是10个），这些模型的训练数据与目标模型不同。多样性的影子模型有助于捕捉到目标模型的不同行为，从而提高攻击模型的泛化能力。
使用标准化：

3.在训练攻击模型时，采用了StandardScaler对攻击数据进行标准化处理。这可以帮助模型更好地学习和适应数据的分布，进一步提升攻击成功率。
简单而有效的攻击模型：

4.采用Logistic Regression作为攻击模型，因为其具有较低的复杂度且易于训练，能够快速有效地学习到攻击特征。

```
### 结果分析

本项目将以经典的图像数据集CIFAR100为例，展示代码的执行过程并分析其输出结果。

首先要进行run_attack.py程序中一些参数和路径的定义，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo16.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo17.png" width="50%">
<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo18.png" width="50%">

开始训练阴影模型，每训练一个阴影模型(这里设置的是20轮)，都会输出类似的信息，展示了该阴影模型训练的损失值，损失值越小就说明训练的越准确，从结果可以看出，随着训练轮数的增多，损失值是呈下降趋势的。

训练所有阴影模型后，继续训练目标模型和攻击模型，并输出模型的准确率，结果如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo19.png" width="40%">

由于CIFAR100数据集的数据量比较大，训练速度会很慢，所以这里使用GPU进行训练，训练的轮数也比较少，读者可以根据环境修改轮数以提高准确率。

从结果可以看出，准确率在0.60以上附近。整体来看，修改攻击方法后准确率有所上升，但模型的表现还有提升的空间，可以进一步优化模型参数和训练策略。

## 复杂场景下的成员推断攻击

### 介绍

该过程主要是在RL模型的基础之上开启复杂场景下的成员推断攻击，并添加一些新的功能代码，其中以经典数据集MNIST为例。

首先，分别对RL模型的训练数据集随机删除5%和10%的数据，记录删除了哪些数据，并分别用剩余数据重新训练RL模型，形成的模型包括原RL模型，删除5%数据后训练的RL模型，删除10%数据后训练的RL模型。然后，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和之后训练而成的模型的攻击成功率。最后，记录攻击对比情况。

上述是完全重训练的方式，即自动化地实现删除，并用剩余数据集训练一个新模型，并保存新模型和原来的模型。本文还采用了其他更快的方法，即分组重训练，具体思路为将数据分组后，假定设置被删除数据落在某一个特定组，重新训练时只需针对该组进行重训练，而不用完全重训练。同样地，保存原来模型和分组重训练后的模型。

### 代码结构
```python
 ├── Complex    # 相关代码目录
 │  ├── Transformer_AI  # VTF模型训练代码
 │      ├── net.py    # Transformer网络模型代码
 │      ├── tf_train.py    # 模型完全重训练代码
 │      ├── tf_part_train.py    # 模型分组重训练代码
 │      └── test.py    # 模型推理代码
 │      └── utils.py    # 模型评估代码
 │      └── dataset.py    # 数据处理代码
 ├  ├── MIA_attack  # 攻击代码
 │      └── net.py    # 网络模型代码
 │      └── run_attack.py   #成员推断攻击代码
 └── README.md 
```

### 实现步骤

1. 首先进行删除数据的操作，定义一个函数remove_random_data，该函数用于从给定的PyTorch数据集中随机删除一定百分比的数据并记录被删除的数据索引，删除后返回删除的数据索引和剩余的数据索引。相关代码如下：
```

    def get_subset_indices(dataset, delete_ratio):
        total_len = len(dataset)
        delete_count = int(total_len * delete_ratio)
        all_indices = list(range(total_len))
        delete_indices = random.sample(all_indices, delete_count)
        remaining_indices = list(set(all_indices) - set(delete_indices))
        return remaining_indices, delete_indices

其中，delect_ratio:要从数据集中删除的数据的百分比，remaining_indices:包含所有未被删除的数据的索引，delect_indices:被删除的数据的索引。

```

特别地，如果要使用分组重训练的方式来训练模型，删除数据的方式和上述相同，只是训练的方式略有不同。
```

for subset_name, delete_indices in zip(["delete_5", "delete_10"], [deleted_indices_5, deleted_indices_10]):
    print(f"\nFine-tuning model with {subset_name} dataset")
    finetune_loader = get_dataloader(delete_indices)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    model.load_state_dict(torch.load("./part_weights_15/model_full.pth"))

```
这部分代码体现了分组重训练的核心思想：对于每个删除数据的子集(5%和10%)，分别加载被删除数据的索引，并以此作为新的训练集进行微调(fine-tuning)。模型从之前保存的完整模型权重开始训练，而不是从头开始训练。


2.然后通过改变delete_ratio的值，生成对未删除数据的数据集、随机删除5%数据后的数据集和随机删除10%数据后的数据集，然后重新训练Transformer模型，形成的模型包括原Transformer模型，删除5%数据后训练的Transformer模型，删除10%数据后训练的Transformer模型。

具体训练步骤与原来无异，区别在于要调用get_subset_indices函数获取不同数据集的索引并记录删除的数据，相关代码如下：
```

full_indices = list(range(len(train_dataset)))
train_indices_5, deleted_indices_5 = get_subset_indices(train_dataset, delete_ratio=0.05)
train_indices_10, deleted_indices_10 = get_subset_indices(train_dataset, delete_ratio=0.10)

注意：如果是在同一个程序中生成用不同数据集训练的模型，要记得在前一个模型训练完之后重新初始化模型，且删除5%和10%数据都是在原数据集的基础上，而不是叠加删除。

```

运行代码后程序会打印出删除%%、10%的样本数据的索引有哪些，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo20.png" width="50%">

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo21.png" width="50%">


```

for epoch in range(args.finetune_epochs):
    train_loss, train_acc = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=finetune_loader,
                                            device=device,
                                            epoch=epoch)
    scheduler.step()
    val_loss, val_acc = evaluate(model=model,
                                 data_loader=val_loader,
                                 device=device,
                                 epoch=epoch)

```
在分组重训练的循环中，模型仅在删除的数据上进行训练。每个小批次的数据集只包含了部分数据(即被删除的数据)，从而实现了对模型的部分更新。

3.利用前面讲到的模型成员攻击算法，分别对上述形成的模型发起成员推断攻击，观察被删除的数据在删除之前和删除之后训练而成的模型的攻击成功率，并记录攻击的对比情况。

具体攻击的方法和步骤和前面讲的差不多，不同点在于，由于这里我们用的训练模型是Transformer模型，所以我们在net.py中要构造这种模型的网络模型。

### 结果记录及分析

1.首先比较删除数据前后Transformer模型的训练准确率，如下图所示：

(1)完全重训练

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo22.png" width="40%">

(图1：未删除数据的Transformer模型训练准确率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo23.png" width="40%">

(图2：删除5%数据后的Transformer训练准确率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo24.png" width="40%">

(图3：删除10%数据后的Transformer训练准确率)

由上述结果可以看出，删除数据后模型训练的平均精确度先是有小幅度的升高，后又小幅度下降。这也说明了数据的数量和模型训练精度的关系不是线性的，它们之间存在复杂的关系，需要更多的尝试来探寻它们之间的联系，而不能一概而论！

(2)分组重训练

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo25.png" width="40%">

(图4：未删除数据的Transformer模型训练准确率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo26.png" width="40%">

(图5：删除5%数据后的Transformer训练准确率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo27.png" width="40%">

(图6：删除10%数据后的Transformer训练准确率)

由于程序会展示出每轮训练所需时间，我们可以明显看出，采用分组重训练的方式，模型训练的速度比完全重训练快得多！这说明，使用分组重训练的方式，可以有效减少训练的时间开销。这个方法被称为增量训练，假设我们可以保存模型状态，并根据删除的数据部分，针对局部数据进行优化。

```
如果删除的数据是噪音数据或outliers，即不具代表性的数据，那么删除这些数据可能会提高模型的精确度。因为这些数据可能会干扰模型的训练，使模型学习到不正确的规律。删除这些数据后，模型可以更好地学习到数据的模式，从而提高精确度。

但是，如果删除的数据是重要的或具代表性的数据，那么删除这些数据可能会降低模型的精确度。因为这些数据可能包含重要的信息，如果删除这些数据，模型可能无法学习到这些信息，从而降低精确度。

此外，删除数据还可能会导致模型的过拟合，即模型过于拟合训练数据，无法泛化到新的数据上。这是因为删除数据后，模型可能会过于依赖剩余的数据，导致模型的泛化能力下降。
```

2.然后开始对形成的模型进行成员推理攻击，首先比较删除数据前后训练而成的Transformer模型的攻击成功率，如下图所示：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo28.png" width="40%">

(图7：未删除数据的Transformer模型攻击成功率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo29.png" width="40%">

(图8：删除5%数据后的Transformer模型攻击成功率)

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo30.png" width="40%">

(图9：删除10%数据后的Transformer模型攻击成功率)

由上述结果可知，随着删除数据的比例增加，模型成员推断攻击的成功率先是有小幅度地升高，然后又有小幅度地降低，但删除10%数据后的RL模型攻击成功率跟不删除数据时的攻击成功率是差不多的，整体的准确率都保持在60%以上。

```
删除一部分数据再进行模型成员推断攻击，攻击的成功率可能会降低。这是因为模型成员推断攻击的原理是利用模型对训练数据的记忆，通过观察模型对输入数据的行为来判断该数据是否在模型的训练集中。

如果删除了一部分数据，模型的训练集就会减少，模型对剩余数据的记忆就会减弱。这样，攻击者就更难以通过观察模型的行为来判断某个数据是否在模型的训练集中，从而降低攻击的成功率。

此外，删除数据还可能会使模型变得更robust，对抗攻击的能力更强。因为模型在训练时需要适应新的数据分布，模型的泛化能力就会提高，从而使攻击者更难以成功地进行成员推断攻击。

但是，需要注意的是，如果删除的数据是攻击者已经知晓的数据，那么攻击的成功率可能不会降低。因为攻击者已经知道这些数据的信息，仍然可以使用这些信息来进行攻击。

本项目所采用的模型都是神经网络类的，如果采用非神经网络类的模型，例如，决策树、K-means等，可能会有不一样的攻击效果，读者可以尝试一下更多类型的模型观察一下。
```

### 文本处理

前面我们实现的Transformer模型都是基于图像数据训练的，接下来我们将引入文本数据，展示Transformer模型在文本处理和文本分类能力上的优势，以扩充Transformer模型的功能。

在训练过程中，首先需要定义训练中需要配置的参数：
```
self.model_name = 'Transformer'
        self.embedding_pretrained = None  # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5 # 随机失活
        self.num_classes = 2  # 类别数
        self.num_epochs = 20  # epoch数
        self.batch_size = 20  # mini-batch大小
        self.pad_size = 500   # 每句话处理成的长度(短填长切)
        self.n_vocab = None#这里需要读取数据的部分进行赋值
        self.learning_rate = 5e-4  # 学习率
        self.embed = 300  # 词向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
        self.checkpoint_path = './model.ckpt'
```
接着我们需要在net.py中添加可以处理文本数据的dataLoader、Transformer模型等，训练集使用常用的IMDB影评数据集，这里默认训练轮数为20轮，读者可以根据需要上述训练参数。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo31.png" width="40%">

训练开始后，程序会输出第一条文本数据："dark comedy? gallows humor? how does one make a comedy out of murder? ..."，这条文本是来自IMDB数据集中的一条影评，经过了去除换行符、转换为小写等处理。输出了该文本在字典编码下的表示：tensor([172, 173, 174, 175, ...])，这段代码将文本转换为词汇表中对应词汇的索引，形成了一个长度为500的张量（按 pad_size=500 截断或填充）。输出显示了这条文本中的每个词的索引。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo32.png" width="40%">

数据加载后，对每条评论进行了 tokenization，将其拆分为词元(token)，并通过词汇表将这些词元转换为对应的索引值。vocab_dict 是通过 torchtext 提供的 vocab 工具构建的，其中包含所有在训练数据中出现过且频次大于5的单词。每个词被分配了一个唯一的索引。

这些信息表示在训练过程中，模型正在处理每个批次(batch)数据，并显示了训练进度。训练数据的批次大小为20，每个epoch处理12500个训练样本。由于数据量较大，训练时间通常会比较长。训练集和测试集的大小在代码中都有详细的打印。训练过程中显示了如下信息：

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo33.png" width="40%">

在训练的最后，代码会绘制出两张图表：
- 损失曲线：显示每个 epoch 后的平均训练损失（loss）。
- 准确率曲线：显示每个 epoch 后的验证集准确率。 这些图表有助于监控训练过程中的性能变化，帮助调整超参数或优化模型。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo34.png" width="40%">

同时，在训练过程中，每次验证集的准确率提升时，都会保存当前训练的最佳模型，这确保了训练过程中模型状态的保存，以便在之后加载和评估。

<img src="https://hepucuncao1.obs.cn-south-1.myhuaweicloud.com/Transformer_AI/photo35.png" width="40%">
