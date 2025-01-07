import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

from graph_model import GCN  # Ensure you have your GCN model defined in this module

# Set seeds for reproducibility
torch.manual_seed(171717)
np.random.seed(171717)

# CONSTANTS
TRAIN_SIZE = 1000
TEST_SIZE = 500
DATASET_NAME = 'Cora'
MODEL_PATH = './gcn_attack_model/'

# Create necessary directories if they don't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Load the dataset
dataset = Planetoid(root='Cora', name=DATASET_NAME, transform=T.NormalizeFeatures())
data = dataset[0]


# Utility function to generate data indices for target and shadow models
def generate_data_indices(data_size, target_size):
    indices = np.arange(data_size)
    target_indices = np.random.choice(indices, target_size, replace=False)
    shadow_indices = np.setdiff1d(indices, target_indices)
    return target_indices, shadow_indices


# Function to randomly delete a percentage of indices
def delete_random_indices(indices, percentage):
    num_to_delete = int(len(indices) * percentage)
    deleted_indices = np.random.choice(indices, num_to_delete, replace=False)
    remaining_indices = np.setdiff1d(indices, deleted_indices)
    return remaining_indices, deleted_indices


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
def train_target_model(percentage):
    target_model = GCN(dataset.num_features, dataset.num_classes)
    train_indices, shadow_indices = generate_data_indices(data.num_nodes, TRAIN_SIZE)
    test_indices, _ = generate_data_indices(data.num_nodes, TEST_SIZE)

    # Randomly delete a percentage of training data
    remaining_train_indices, deleted_train_indices = delete_random_indices(train_indices, percentage)

    print(f"Deleted indices for {int(percentage * 100)}% deletion: {deleted_train_indices}")

    train_acc, test_acc, target_model, logits = train_gcn_model(target_model, data, remaining_train_indices,
                                                                test_indices)
    print(f"Target model - Train accuracy (after deletion): {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

    attack_x, attack_y, classes = prepare_attack_data(target_model, data, remaining_train_indices, test_indices)

    # Save attack data for training the attack model
    np.savez(MODEL_PATH + f'attack_data_{int(percentage * 100)}%.npz', attack_x=attack_x, attack_y=attack_y,
             classes=classes)
    return attack_x, attack_y, classes


# Train the shadow models and prepare data for the attack model
def train_shadow_models(n_shadow=10, percentage=0.0):
    shadow_attack_x, shadow_attack_y, shadow_classes = [], [], []

    for i in range(n_shadow):
        shadow_model = GCN(dataset.num_features, dataset.num_classes)
        shadow_train_indices, shadow_test_indices = generate_data_indices(data.num_nodes, TRAIN_SIZE)

        # Randomly delete a percentage of shadow training data
        remaining_shadow_train_indices, deleted_shadow_train_indices = delete_random_indices(shadow_train_indices,
                                                                                             percentage)

        print(
            f"Shadow model {i + 1} deleted indices for {int(percentage * 100)}% deletion: {deleted_shadow_train_indices}")

        train_acc, test_acc, shadow_model, logits = train_gcn_model(shadow_model, data, remaining_shadow_train_indices,
                                                                    shadow_test_indices)
        print(f"Shadow model {i + 1} - Train accuracy (after deletion): {train_acc:.4f}, Test accuracy: {test_acc:.4f}")

        attack_x, attack_y, classes = prepare_attack_data(shadow_model, data, remaining_shadow_train_indices,
                                                          shadow_test_indices)
        shadow_attack_x.append(attack_x)
        shadow_attack_y.append(attack_y)
        shadow_classes.append(classes)

    shadow_attack_x = np.vstack(shadow_attack_x)
    shadow_attack_y = np.concatenate(shadow_attack_y)
    shadow_classes = np.concatenate(shadow_classes)

    np.savez(MODEL_PATH + f'shadow_attack_data_{int(percentage * 100)}%.npz', shadow_attack_x=shadow_attack_x,
             shadow_attack_y=shadow_attack_y, shadow_classes=shadow_classes)
    return shadow_attack_x, shadow_attack_y, shadow_classes


# Train the attack model using the prepared data
def train_attack_model(attack_data=None):
    if attack_data is None:
        attack_data = np.load(MODEL_PATH + 'attack_data.npz')
        attack_x, attack_y, _ = attack_data['attack_x'], attack_data['attack_y'], attack_data['classes']
    else:
        attack_x, attack_y = attack_data

    # Standardize data
    scaler = StandardScaler()
    attack_x = scaler.fit_transform(attack_x)

    # Use RandomForestClassifier for attack model
    attack_model = RandomForestClassifier(n_estimators=100)
    attack_model.fit(attack_x, attack_y)

    # Evaluate the attack model
    pred_y = attack_model.predict(attack_x)
    acc = accuracy_score(attack_y, pred_y)
    print(f"Attack model accuracy: {acc:.4f}")

    # Handle undefined precision with zero_division=0
    print(classification_report(attack_y, pred_y, zero_division=0))


if __name__ == "__main__":
    percentages = [0.05, 0.10]  # Define the deletion percentages

    for percentage in percentages:
        print(f"\nTraining with {int(percentage * 100)}% deletion:")

        # Train the target model and prepare attack data
        attack_x, attack_y, classes = train_target_model(percentage)

        # Train shadow models and prepare shadow attack data
        shadow_attack_x, shadow_attack_y, shadow_classes = train_shadow_models(n_shadow=10, percentage=percentage)

        # Train and evaluate the attack model
        train_attack_model((shadow_attack_x, shadow_attack_y))
