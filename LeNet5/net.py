import numpy as np
import common
from collections import OrderedDict
import pickle
from torchvision import datasets,transforms #导入数据集

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.Resize((28, 28))  # Resize to 28x28
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.Resize((28, 28))  # Resize to 28x28
])
#定义一个网络模型
class LeNet5():
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num1': 6, 'filter_size1': 3, 'filter_num2': 16, 'filter_size2': 3, 'pad': 1,
                             'stride': 1},
                 hidden_size1=120, hidden_size2=84, output_size=10, weight_init_std=0.01):
        filter_num1 = conv_param['filter_num1']
        filter_size1 = conv_param['filter_size1']
        filter_num2 = conv_param['filter_num2']
        filter_size2 = conv_param['filter_size2']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size1 = (input_size - filter_size1 + 2 * filter_pad) / filter_stride + 1
        pool_output_size1 = (conv_output_size1 - 2) / 2 + 1
        conv_output_size2 = (pool_output_size1 - filter_size2 + 2 * filter_pad) / filter_stride + 1
        pool_output_size2 = int((((conv_output_size2 - 2) / 2 + 1) ** 2) * filter_num2)

        #权重初始化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num1, input_dim[0], filter_size1, filter_size1)
        self.params['b1'] = np.zeros(filter_num1)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num2, filter_num1, filter_size2, filter_size2)
        self.params['b2'] = np.zeros(filter_num2)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(pool_output_size2, hidden_size1)
        self.params['b3'] = np.zeros(hidden_size1)
        self.params['gamma1'] = weight_init_std * \
                                np.random.randn(hidden_size1)
        self.params['beta1'] = np.zeros(hidden_size1)
        self.params['W4'] = weight_init_std * \
                            np.random.randn(hidden_size1, hidden_size2)
        self.params['b4'] = np.zeros(hidden_size2)
        self.params['gamma2'] = weight_init_std * \
                                np.random.randn(hidden_size2)
        self.params['beta2'] = np.zeros(hidden_size2)
        self.params['W5'] = weight_init_std * \
                            np.random.randn(hidden_size2, output_size)
        self.params['b5'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()
        self.layers['Conv1'] = common.Convolution(self.params['W1'], self.params['b1'],
                                                  conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = common.Relu()
        self.layers['Pool1'] = common.MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Conv2'] = common.Convolution(self.params['W2'], self.params['b2'],
                                                  conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = common.Relu()
        self.layers['Pool2'] = common.MaxPooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = common.Affine(self.params['W3'], self.params['b3'])
        self.layers['BN1'] = common.BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu3'] = common.Relu()
        self.layers['Affine2'] = common.Affine(self.params['W4'], self.params['b4'])
        self.layers['BN2'] = common.BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu4'] = common.Relu()
        self.layers['Affine3'] = common.Affine(self.params['W5'], self.params['b5'])

        self.last_layer = common.SoftmaxWithCrossEntropy()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        # 从独热编码转回数字编码
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['gamma1'], grads['beta1'] = self.layers['BN1'].dgamma, self.layers['BN1'].dbeta
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['gamma2'], grads['beta2'] = self.layers['BN2'].dgamma, self.layers['BN2'].dbeta
        grads['W5'], grads['b5'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads

    # 只保留权重信息，不包含网络模型
    def save_params(self, file_name="best_model.pth"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="best_model.pth"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Affine1', 'Affine2', 'Affine3']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

if __name__=="__main__":
    x=np.random.rand(1,1,28,28)
    model=LeNet5()  #把网络实例化
