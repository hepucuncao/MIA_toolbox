import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import utils

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        
        layers = nn.ModuleList()

        conv_layer = []
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU(inplace=True))

        self.layers = layers
            
    def forward(self, x):
        fwd = self.layers[0](x) 
        fwd += self.layers[1](x) 
        fwd = self.layers[2](fwd) 
        return fwd

    
class ResNet(nn.Module):
    def __init__(self, args, params):
        super(ResNet, self).__init__()
        self.num_blocks = params['num_blocks']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type']
        #self.port_num = params['port_num']
        self.train_func = utils.cnn_train
        self.test_func = utils.cnn_test
        self.in_channels = 16
        self.num_output =  1

        if self.block_type == 'basic':
            self.block = BasicBlock

        init_conv = []
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))

        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(self.in_channels, block_id=0, stride=1))
        self.layers.extend(self._make_layer(32, block_id=1, stride=2))
        self.layers.extend(self._make_layer(64, block_id=2, stride=2))
        
        end_layers = []
        #if self.input_size > 32:
        #    end_layers.append(nn.AvgPool2d(kernel_size=8))  # 增大 kernel_size 防止过度池化
        #else:
        #    end_layers.append(nn.AvgPool2d(kernel_size=4))  # 适当修改大小
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(utils.Flatten())
        end_layers.append(nn.Linear(64*self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

        self.augment_training = params['augment_training']
        if 'distill' in args.mode:
            self.train_func = utils.cnn_train_dis
        else:
            self.train_func = utils.cnn_train
        self.test_func = utils.cnn_test

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self.block(self.in_channels, channels, stride))
            self.in_channels = channels * self.block.expansion
        return layers

    def forward(self, x):
        out = self.init_conv(x)
        
        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# LeNet 模型
class LeNet(nn.Module):
    def __init__(self, args, params):
        super(LeNet, self).__init__()
        # 使用 params 中的参数来配置模型
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.block_type = params['block_type'] 
        self.train_func = utils.cnn_train
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 通过一次前向传播，动态计算展平后的尺寸
        self._to_linear = None
        self._get_to_linear(self.input_size)

        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def _get_to_linear(self, input_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)  # 假设输入图像是 3 通道
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            self._to_linear = x.numel()  # 得到展平后的尺寸

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平操作
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNN(nn.Module):
    def __init__(self, args, params):
        super(RNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 使用 params 中的参数来配置模型
        self.num_classes = int(params['num_classes'])
        self.block_type = params['block_type']
        self.train_func = utils.cnn_train
        
        # 设定隐藏层大小和层数
        self.hidden_size = 128
        self.num_layers = 2

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)

        # 先定义 RNN，但不硬编码 input_size
        self.rnn = None
        self.fc = nn.Linear(self.hidden_size, self.num_classes).to(self.device)

    def _get_rnn_input_size(self, input_size):
        """动态计算卷积层输出的尺寸，用于 RNN 输入"""
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size).to(self.device)  # 假设输入图像是 3 通道
            x = self.pool(torch.relu(self.conv1(x)))  # 通过卷积和池化层
            batch_size, channels, height, width = x.size()
            seq_len = height
            input_size = channels * width
            return seq_len, input_size

    def forward(self, x):
        # 确保输入数据与模型参数在同一设备上
        x = x.to(self.device)  # 将输入数据移动到设备上
        
        # 获取卷积输出的尺寸，计算 RNN 的输入大小
        seq_len, input_size = self._get_rnn_input_size(x.size(2))  # x 的形状是 [batch_size, channels, height, width]
        
        # 定义 RNN 层
        if self.rnn is None:  # 只初始化一次 RNN
            self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True).to(self.device)
        
        # 通过卷积层提取特征
        x = self.pool(torch.relu(self.conv1(x)))  # 输出形状: [batch_size, 16, H, W]
        
        # 动态计算卷积输出的尺寸
        batch_size, channels, height, width = x.size()
        
        # 将 x 变形为适合 RNN 的输入: [batch_size, seq_len, input_size]
        x = x.view(batch_size, seq_len, input_size)

        # 传递给 RNN
        out, _ = self.rnn(x)  # out 的形状: [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]  # 获取最后一个时间步的输出
        out = self.fc(out)  # 最后一个全连接层
        return out


class RL(nn.Module):
    def __init__(self, args, params):
        super(RL, self).__init__()

        # 从 params 字典动态获取所需的参数
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.port_num = int(params['port_num'])
        
        self.input_size = self.input_size * self.input_size * self.port_num  # 展平后的输入尺寸
        self.hidden_dim = 64  # 隐藏层维度
        self.output_dim = self.num_classes  # 输出维度等于类别数
        self.train_func = utils.cnn_train
        
        # fc_layer 的输入维度应与输入数据的维度匹配
        self.fc_layer = nn.Linear(self.input_size, self.hidden_dim)  # 输入大小改为3072
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平：batch_size x input_size
        x = self.fc_layer(x)  # 输入到全连接层
        x = self.relu(x)  # 激活
        x = self.output_layer(x)  # 输出层
        return x


    def get_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            action = torch.argmax(action_probs).item()
            return action

    def get_action_probs(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            action_probs = self.forward(state)
            action_probs = torch.softmax(action_probs, dim=1)
            return action_probs

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            value = self.forward(state)
            return value



    
class wide_basic(nn.Module):
    def __init__(self, in_channels, channels, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.layers = nn.ModuleList()
        conv_layer = []
        conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=True))
        conv_layer.append(nn.Dropout(p=dropout_rate))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU(inplace=True))
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=True))

        self.layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=True),
            )

        self.layers.append(shortcut)

    def forward(self, x):
        out = self.layers[0](x)
        out += self.layers[1](x)
        return out

class WideResNet(nn.Module):
    def __init__(self, args, params):
        super(WideResNet, self).__init__()
        self.num_blocks = params['num_blocks']
        self.widen_factor = params['widen_factor']
        self.num_classes = int(params['num_classes'])
        self.dropout_rate = params['dropout_rate']
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.augment_training = params['augment_training']
        if 'distill' in args.mode:
            self.train_func = utils.cnn_train_dis
        else:
            self.train_func = utils.cnn_train
        self.test_func = utils.cnn_test
        self.in_channels = 16
        self.num_output =  1

        self.init_conv = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            
        self.layers = nn.ModuleList()
        self.layers.extend(self._wide_layer(wide_basic, self.in_channels*self.widen_factor, block_id=0, stride=1))
        self.layers.extend(self._wide_layer(wide_basic, 32*self.widen_factor, block_id=1, stride=2))
        self.layers.extend(self._wide_layer(wide_basic, 64*self.widen_factor, block_id=2, stride=2))

        end_layers = []

        end_layers.append(nn.BatchNorm2d(64*self.widen_factor, momentum=0.9))
        end_layers.append(nn.ReLU(inplace=True))
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(utils.Flatten())
        end_layers.append(nn.Linear(64*self.widen_factor, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)


        self.initialize_weights()

    def _wide_layer(self, block, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, self.dropout_rate, stride))
            self.in_channels = channels
        return layers

    def forward(self, x):
        out = self.init_conv(x)

        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, conv_params):
        super(ConvBlock, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        avg_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))
                
        conv_layers.append(nn.ReLU())
                
        if avg_pool_size > 1:
            conv_layers.append(nn.AvgPool2d(kernel_size=avg_pool_size))
        
        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class FcBlock(nn.Module):
    def __init__(self, fc_params, flatten):
        super(FcBlock, self).__init__()
        input_size = int(fc_params[0])
        output_size = int(fc_params[1])

        fc_layers = []
        if flatten:
            fc_layers.append(utils.Flatten())
        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))        
        self.layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class VGG(nn.Module):
    def __init__(self, args, params):
        super(VGG, self).__init__()

        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels'] 
        self.fc_layer_sizes = params['fc_layers']

        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.init_weights = params['init_weights']
        self.augment_training = params['augment_training']
        if 'distill' in args.mode:
            self.train_func = utils.cnn_train_dis
        else:
            self.train_func = utils.cnn_train
        self.test_func = utils.cnn_test
        self.num_output = 1

        self.init_conv = nn.Sequential()

        self.layers = nn.ModuleList()
        input_channel = 3
        cur_input_size = self.input_size
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            self.layers.append(ConvBlock(conv_params))
            input_channel = channel
        
        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True
            
            self.layers.append(FcBlock(fc_params, flatten=flatten))
            fc_input_size = width
        
        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def forward(self, x):
        fwd = self.init_conv(x)

        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd

class MobileNet(nn.Module):
    def __init__(self, args,params):
        super(MobileNet, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        if 'distill' in args.mode:
            self.train_func = utils.cnn_train_dis
        else:
            self.train_func = utils.cnn_train
        self.test_func = utils.cnn_test
        self.num_output = 1
        self.in_channels = 32
        init_conv = []
        
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []

        end_layers.append(nn.AvgPool2d(2))

        end_layers.append(utils.Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        return layers

    def forward(self, x):
        fwd = self.init_conv(x)
        for layer in self.layers:
            fwd = layer(fwd)

        fwd = self.end_layers(fwd)
        return fwd



