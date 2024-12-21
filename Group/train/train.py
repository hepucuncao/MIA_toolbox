import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from net import GCN
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import StepLR
import os

# 选择数据集（Cora、CiteSeer、PubMed）
dataset_name = 'Cora'  # 可以换成 'CiteSeer' 或 'PubMed'
dataset = Planetoid(root=f'Cora', name='Cora', transform=NormalizeFeatures())

# 模型参数
input_dim = dataset.num_node_features
hidden_dim = 128
output_dim = dataset.num_classes

# 加载数据
data = dataset[0]

# 创建模型
model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

# 训练和评估
best_test_acc = 0
best_model_state = None  # 用于存储最佳模型的状态字典
for epoch in range(1, 501):
    loss = train()
    train_acc, val_acc, test_acc = test()

    if test_acc > best_test_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_model_state = model.state_dict()  # 更新最佳模型状态

    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# 保存最佳模型参数
folder = 'save_model'
if not os.path.exists(folder):
    os.mkdir('save_model')
print('save best model')
if best_model_state:
    torch.save(best_model_state, 'save_model/best_model.pth')

print(f'Best Test Accuracy: {best_test_acc:.4f}')
