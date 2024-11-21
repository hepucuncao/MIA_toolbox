# 支持数据撤销的模型成员推理攻击工具箱

本项目是一个支持数据撤销的成员推理攻击工具，支持mnist、cifar10、cifar100、cinic10等多种数据集，可实现针对LeNet、RNN、ResNet、RL等多种模型的成员推理攻击，并给用户一个详细的使用文档。
## 环境

 - Python 3.9

 - PyTorch>=1.10.0

## 用法

### 第一步：安装并导入该库

在mia-1.0-py3-none-any.whl文件路径下，运行以下命令，就可以将whl文件库安装到Python环境中了，从而可以通过import将该库导入。
```
pip install mia-1.0-py3-none-any.whl
```

### 第二步：训练目标模型和影子模型

在你的代码中通过import mia就可以导入该库，或通过from mia import core,utils导入必要的模块，并需要传入必要的参数，可以在程序中定义默认值也可以通过命令行传入，接着就可以调用core中的函数来进行成员推理攻击。其中，detected和ration参数分别代表是否启动数据撤销功能以及撤销数据的比例，默认不进行数据撤销。下面展示命令行传入的过程：
- 训练目标模型
```
python test.py --mode target
```
- 训练影子模型
```
python test.py --mode shadow
```
影子模型经过训练以模拟目标模型的行为。

### 第二步（可选）：Distill目标模型和影子模型

- Distill目标模型
```
python test.py --mode distill_target
```
- Distill影子模型
```
python test.py --mode distill_shadow
```

### 第三步：为攻击模型准备数据集

- 获取攻击模型训练数据
```
python test.py --action 1 --mode shadow --mia_type build-dataset
```
- 获取影子模型训练数据
```
python test.py --action 1 --mode target --mia_type build-dataset
```

### 第四步：训练和测试攻击模型

```
python main.py --action 1 --mia_type black-box
```
训练的模型和生成的数据将保存到 './networks/{seed}/{mode}/{data}_{model}'。

## 实例

本项目默认使用CIFAR100数据集，针对ResNet56模型且epoch为100，您可以根据需要更改参数。

下面是一个使用该库的示例代码(demo.py)：
```
import argparse
from mia import core,utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIA_toolbox')
    parser.add_argument('--action', type=int, default=0, help=[0, 1, 2])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target',
                        help=['target', 'shadow', 'distill_target', 'distill_shadow'])
    parser.add_argument('--model', type=str, default='resnet',
                        help=['resnet', 'mobilenet', 'vgg', 'wideresnet', 'lenet', 'rnn', 'rl'])
    parser.add_argument('--data', type=str, default='cifar100',
                        help=['cinic10', 'cifar10', 'cifar100', 'gtsrb', 'mnist'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model_distill', type=str, default='resnet',
                        help=['resnet', 'mobilenet', 'vgg', 'wideresnet', 'lenet', 'rnn', 'rl'])
    parser.add_argument('--epochs_distill', type=int, default=100)
    parser.add_argument('--mia_type', type=str, help=['build-dataset', 'black-box'])
    parser.add_argument('--port_num', type=int, default=3)
    parser.add_argument('--is_detected', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.05)

    args = parser.parse_args()
    if args.action == 0:
        core.train_networks(args)

    elif args.action == 1:
        core.membership_inference_attack(args)


if __name__ == "__main__":
    main()
```
上述代码中的参数可以根据需求删除修改，没有添加的参数使用默认参数，在demo.py目录下，在命令行中依次输入以下命令：
```
python demo.py --mode target --detected 1 
python demo.py --mode shadow
python demo.py --mode distill_target
python demo.py --mode distill_shadow
python demo.py --action 1 --mode shadow --mia_type build-dataset
python demo.py --action 1 --mode target --mia_type build-dataset
python demo.py --action 1 --mia_type black-box
```
数据集默认下载在c01yili目录下，没有的数据集会自动下载；

训练的模型和生成的数据将保存到'./networks/{seed}/{mode}/{cifar100}_{resnet56}'；

训练和攻击的结果将默认保存到'./outputs/train_models.out，并保存训练好的模型cifar100_resnet_resnet_trajectory_auc.npy

## 引文

有关技术细节和完整的实验结果，请参阅以下论文。
```
@inproceedings{LZBZ22,
author = {Yiyong Liu and Zhengyu Zhao and Michael Backes and Yang Zhang},
title = {{Membership Inference Attacks by Exploiting Loss Trajectory}},
booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
pages = {2085-2098},
publisher = {ACM},
year = {2022}
}
```

## 联系

如果您对代码有任何疑问，请随机联系2319128705@qq.com
