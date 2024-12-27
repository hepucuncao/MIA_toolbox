import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一层卷积和ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 使用Dropout防止过拟合
        x = F.dropout(x, p=0.6, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 输出分类结果
        return F.log_softmax(x, dim=1)


