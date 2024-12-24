import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
import os
from graph_model import GCN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


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


# Set seeds for reproducibility
torch.manual_seed(171717)
np.random.seed(171717)

# CONSTANTS
TRAIN_SIZE = 1000
TEST_SIZE = 500
DATASET_NAME = 'PubMed'
MODEL_PATH = './gcn_attack_model/'

# Load the dataset
dataset = Planetoid(root='PubMed', name=DATASET_NAME, transform=T.NormalizeFeatures())
data = dataset[0]

# Create necessary directories if they don't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Utility function to generate data indices for target and shadow models
def generate_data_indices(data_size, target_size):
    indices = np.arange(data_size)
    target_indices = np.random.choice(indices, target_size, replace=False)
    shadow_indices = np.setdiff1d(indices, target_indices)
    return target_indices, shadow_indices


# Function to train the GCN model
def train_gcn_model(model, data, train_indices, test_indices, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
    return train_acc, test_acc, model, out


# Function to prepare attack data (features from the model and labels for MIA)
def prepare_attack_data(model, data, train_indices, test_indices):
    model.eval()
    with torch.no_grad():
        logits = model(data)

    attack_x, attack_y, classes = [], [], []

    # Prepare data from the training set (labeled as "1" for being part of training)
    for idx in train_indices:
        attack_x.append(logits[idx].cpu().numpy())
        attack_y.append(1)
        classes.append(data.y[idx].item())

    # Prepare data from the test set (labeled as "0" for not being part of training)
    for idx in test_indices:
        attack_x.append(logits[idx].cpu().numpy())
        attack_y.append(0)
        classes.append(data.y[idx].item())

    attack_x = np.array(attack_x, dtype=np.float32)
    attack_y = np.array(attack_y, dtype=np.int32)
    classes = np.array(classes, dtype=np.int32)

    return attack_x, attack_y, classes


# Train the target model and prepare data for the attack model
def train_target_model():
    target_model = GCN(dataset.num_features, dataset.num_classes)
    train_indices, shadow_indices = generate_data_indices(data.num_nodes, TRAIN_SIZE)
    test_indices, _ = generate_data_indices(data.num_nodes, TEST_SIZE)

    train_acc, test_acc, target_model, logits = train_gcn_model(target_model, data, train_indices, test_indices)
    print(f"Target model - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    attack_x, attack_y, classes = prepare_attack_data(target_model, data, train_indices, test_indices)

    # Save attack data for training the attack model
    np.savez(MODEL_PATH + 'attack_data.npz', attack_x=attack_x, attack_y=attack_y, classes=classes)
    return attack_x, attack_y, classes


# Train the shadow models and prepare data for the attack model
def train_shadow_models(n_shadow=10):
    shadow_attack_x, shadow_attack_y, shadow_classes = [], [], []

    for i in range(n_shadow):
        shadow_model = GCN(dataset.num_features, dataset.num_classes)
        shadow_train_indices, shadow_test_indices = generate_data_indices(data.num_nodes, TRAIN_SIZE)

        train_acc, test_acc, shadow_model, logits = train_gcn_model(shadow_model, data, shadow_train_indices,
                                                                    shadow_test_indices)
        print(f"Shadow model {i + 1} - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        attack_x, attack_y, classes = prepare_attack_data(shadow_model, data, shadow_train_indices, shadow_test_indices)
        shadow_attack_x.append(attack_x)
        shadow_attack_y.append(attack_y)
        shadow_classes.append(classes)

    shadow_attack_x = np.vstack(shadow_attack_x)
    shadow_attack_y = np.concatenate(shadow_attack_y)
    shadow_classes = np.concatenate(shadow_classes)

    np.savez(MODEL_PATH + 'shadow_attack_data.npz', shadow_attack_x=shadow_attack_x, shadow_attack_y=shadow_attack_y,
             shadow_classes=shadow_classes)
    return shadow_attack_x, shadow_attack_y, shadow_classes


# Train the attack model using the prepared data
def train_attack_model(attack_data=None, epochs=500, batch_size=64, learning_rate=0.001):
    np.savez(MODEL_PATH + 'shadow_attack_data.npz', shadow_attack_x=shadow_attack_x, shadow_attack_y=shadow_attack_y,
             shadow_classes=shadow_classes)

    if attack_data is None:
        attack_data = np.load(MODEL_PATH + 'attack_data.npz')
        attack_x, attack_y, _ = attack_data['attack_x'], attack_data['attack_y'], attack_data['classes']
    else:
        attack_x, attack_y = attack_data

    # 标准化数据
    scaler = StandardScaler()
    attack_x = scaler.fit_transform(attack_x)

    # Train attack model (a simple logistic regression)
    attack_model = LogisticRegression(max_iter=epochs, solver='lbfgs')
    attack_model.fit(attack_x, attack_y)

    # Evaluate the attack model
    pred_y = attack_model.predict(attack_x)
    acc = accuracy_score(attack_y, pred_y)
    print(f"Attack model accuracy: {acc:.4f}")

    # Handle undefined precision with zero_division=0
    print(classification_report(attack_y, pred_y, zero_division=0))


if __name__ == "__main__":
    # Train the target model and prepare attack data
    attack_x, attack_y, classes = train_target_model()

    # Train shadow models and prepare shadow attack data
    shadow_attack_x, shadow_attack_y, shadow_classes = train_shadow_models(n_shadow=10)

    # Train and evaluate the attack model
    train_attack_model((shadow_attack_x, shadow_attack_y))
