import torch
import torch_geometric
import torch.nn.functional as F

# Define GCN model class (assuming 2-layer GCN)
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_features, 16)
        self.conv2 = torch_geometric.nn.GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
