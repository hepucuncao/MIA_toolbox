import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from net import GCN
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np

# 选择数据集（Cora、CiteSeer、PubMed）
dataset_name = 'CiteSeer'  # 可以换成 'CiteSeer' 或 'PubMed'
dataset = Planetoid(root=f'CiteSeer', name='CiteSeer', transform=NormalizeFeatures())

# 模型参数
input_dim = dataset.num_node_features
hidden_dim = 256  # 增加隐藏层维度
output_dim = dataset.num_classes

# 加载数据
data = dataset[0]


# 创建模型
def create_model():
    return GCN(input_dim, hidden_dim, output_dim)


# 更新优化器
def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 降低学习率和权重衰减


# 训练函数
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试函数
def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs


# 删除数据并记录索引
def delete_random_nodes(data, percentage):
    num_nodes = data.num_nodes
    num_to_delete = int(num_nodes * percentage)
    all_indices = np.arange(num_nodes)
    np.random.shuffle(all_indices)
    indices_to_delete = all_indices[:num_to_delete]

    # 创建新的 train_mask
    new_train_mask = data.train_mask.clone()
    new_train_mask[indices_to_delete] = False  # 删除指定节点
    return new_train_mask, indices_to_delete


# 保存模型的函数
def save_model(model, folder, filename):
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(model.state_dict(), os.path.join(folder, filename))


# 训练和评估
def run_experiment(deletion_percentage, suffix):
    # 修改 train_mask
    new_train_mask, deleted_indices = delete_random_nodes(data, deletion_percentage)
    data.train_mask = new_train_mask

    print(
        f'\nTraining GCN model with {int(deletion_percentage * 100)}% data deletion. Deleted nodes: {deleted_indices.tolist()}')

    model = create_model()  # 新建模型实例
    optimizer = get_optimizer(model)  # 获取优化器

    best_test_acc = 0
    best_model_state = None  # 用于存储最佳模型的状态字典
    for epoch in range(1, 501):  # 增加训练周期
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)

        if test_acc > best_test_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_state = model.state_dict()  # 更新最佳模型状态

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 保存最佳模型参数
    save_model(model, 'save_model', f'best_model_{suffix}.pth')
    print(f'Best Test Accuracy with {int(deletion_percentage * 100)}% deletion: {best_test_acc:.4f}')


# 训练原始模型
run_experiment(0.0, 'original')

# 训练删除 5% 数据后的模型
run_experiment(0.05, '5_percent_deleted')

# 训练删除 10% 数据后的模型
run_experiment(0.10, '10_percent_deleted')
