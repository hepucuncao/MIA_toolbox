import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from net import GCN
import numpy as np
import os

# 选择数据集（Cora、CiteSeer、PubMed）
dataset_name = 'Cora'  # 可以换成 'CiteSeer' 或 'PubMed'
dataset = Planetoid(root=f'Cora', name='Cora', transform=NormalizeFeatures())

# 模型参数
input_dim = dataset.num_node_features
hidden_dim = 256  # 隐藏层维度
output_dim = dataset.num_classes

# 加载数据
data = dataset[0]


# 创建模型
def create_model():
    return GCN(input_dim, hidden_dim, output_dim)


# 更新优化器
def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


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


# 保存模型的函数
def save_model(model, folder, filename):
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save(model.state_dict(), os.path.join(folder, filename))


# 分组函数
def group_data(data, num_groups):
    nodes_per_group = data.num_nodes // num_groups
    groups = [np.arange(i * nodes_per_group, (i + 1) * nodes_per_group) for i in range(num_groups)]

    # 处理可能存在的余数节点
    if data.num_nodes % num_groups != 0:
        groups[-1] = np.concatenate((groups[-1], np.arange(num_groups * nodes_per_group, data.num_nodes)))

    return np.array(groups, dtype=object)  # 使用 dtype=object 确保可以存储不同长度的数组


# 删除随机节点
def delete_random_nodes(data, percentage):
    num_nodes = data.num_nodes
    num_to_delete = int(num_nodes * percentage)
    all_indices = np.arange(num_nodes)
    np.random.shuffle(all_indices)
    indices_to_delete = all_indices[:num_to_delete]
    return indices_to_delete


# 训练原始模型
def run_full_training():
    model = create_model()
    optimizer = get_optimizer(model)
    best_test_acc = 0
    best_model_state = None

    for epoch in range(1, 501):
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    save_model(model, 'save_model', 'best_model_full.pth')
    print(f'Best Test Accuracy (Full Training): {best_test_acc:.4f}')


# 分组重训练
def run_group_retraining(deletion_percentage, num_groups):
    groups = group_data(data, num_groups)
    deleted_group = groups[np.random.randint(num_groups)]  # 随机选择一个组
    indices_to_delete = delete_random_nodes(data, deletion_percentage)

    print(f'\nTraining GCN model with {int(deletion_percentage * 100)}% data deletion in group {deleted_group}.')

    # 创建新模型并获取优化器
    model = create_model()
    optimizer = get_optimizer(model)

    # 仅更新被删除组的训练掩码
    new_train_mask = data.train_mask.clone()
    new_train_mask[indices_to_delete] = False

    for epoch in range(1, 501):  # 只训练被影响的组
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    save_model(model, 'save_model', f'best_model_group_{int(deletion_percentage * 100)}.pth')


# 运行完整训练
run_full_training()

# 运行分组重训练，分别删除 5% 和 10% 数据
run_group_retraining(0.05, num_groups=5)
run_group_retraining(0.10, num_groups=5)
