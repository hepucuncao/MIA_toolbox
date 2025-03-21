import torch
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from net import GCN
import matplotlib.pyplot as plt

# 选择数据集（Cora、CiteSeer、PubMed等）
dataset_name = 'Cora'
dataset = Planetoid(root=f'Cora', name='Cora', transform=NormalizeFeatures())

# 模型参数
input_dim = dataset.num_node_features
hidden_dim = 128
output_dim = dataset.num_classes

# 加载数据
data = dataset[0]

# 创建模型
model = GCN(input_dim, hidden_dim, output_dim)

# 加载保存的模型参数
model.load_state_dict(torch.load("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/Graph/save_model/best_model.pth"))

# 模型推理时切换到评估模式
model.eval()

# 模型推理
def inference():
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
        out = model(data)
        pred = out.argmax(dim=1)  # 预测类别
    return pred

# 进行推理测试
def test(num_samples=20):
    pred = inference()

    # 获取训练集实际标签和预测标签
    train_mask = data.train_mask
    actual_labels = data.y[train_mask]  #实际标签
    predicted_labels = pred[train_mask]  #预测标签

    # 随机选择 num_samples 个样本的索引
    sample_indices = random.sample(range(train_mask.sum().item()), num_samples)

    # 打印实际标签与预测标签
    for sample_index in sample_indices:
        idx = train_mask.nonzero(as_tuple=True)[0][sample_index]  # 获取训练集中样本的实际索引
        actual = actual_labels[idx].item()
        predicted = predicted_labels[idx].item()
        print(f'Actual: {actual}, Predicted: {predicted}, Match: {actual == predicted}')

        # 可视化节点特征
        feature = data.x[idx].numpy().reshape(1, -1)  # 将特征重塑为1x特征数
        plt.imshow(feature, cmap='gray', aspect='auto')
        plt.title(f'Actual: {actual}, Predicted: {predicted}')
        plt.axis('off')
        plt.show()


# 进行推理测试
train_acc = test(num_samples=20)

