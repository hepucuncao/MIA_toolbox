import os
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import vit_base_patch16_224_in21k as create_model

def load_data():
    # 加载 CIFAR-100 数据集
    trainset = datasets.CIFAR100(root='cifar100', train=True, download=True, transform=None)
    testset = datasets.CIFAR100(root='cifar100', train=False, download=True, transform=None)

    return trainset, testset

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载训练和测试数据集
    train_dataset, test_dataset = load_data()

    # 对数据集应用变换
    train_dataset.transform = data_transform
    test_dataset.transform = data_transform

    # 创建数据加载器
    batch_size = 16  # 设置batch size
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers)

    # 加载类别索引
    class_indices = {i: label for i, label in enumerate(train_dataset.classes)}
    with open('class_indices.json', 'w') as json_file:
        json.dump(class_indices, json_file, indent=4)

    # 加载模型
    model = create_model(num_classes=100, has_logits=False).to(device)
    model_weight_path = "./weights_15/model-14.pth"
    assert os.path.exists(model_weight_path), f"Model weights '{model_weight_path}' not found."
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 执行测试集评估
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt="", cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate precision, recall, F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions)

    # Plot precision, recall, F1 score for each class
    plot_metrics_per_class(precision, recall, f1, class_indices)

    # Calculate and plot weighted precision, recall, F1 score
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    plot_metrics(precision_weighted, recall_weighted, f1_weighted)

    # Display classification report
    report = classification_report(all_targets, all_predictions, target_names=list(class_indices.values()), digits=4)
    print(report)

    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_targets) * 100
    print(f"Accuracy: {accuracy:.2f}%")


def plot_metrics_per_class(precision, recall, f1, class_indices):
    metrics = ['Precision', 'Recall', 'F1 Score']
    num_classes = len(class_indices)
    x_pos = np.arange(num_classes)

    plt.figure(figsize=(20, 10))

    plt.bar(x_pos - 0.2, precision, 0.2, label='Precision', color='blue')
    plt.bar(x_pos, recall, 0.2, label='Recall', color='green')
    plt.bar(x_pos + 0.2, f1, 0.2, label='F1 Score', color='orange')

    plt.xticks(x_pos, list(class_indices.values()), rotation=90)
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score per Class')
    plt.ylim(0, 1.1)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_metrics(precision, recall, f1):
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    x_pos = np.arange(len(metrics))

    plt.figure(figsize=(8, 6))
    plt.bar(x_pos, values, align='center', alpha=0.8, color=['blue', 'green', 'orange'])
    plt.xticks(x_pos, metrics)
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score')
    plt.ylim(0, 1.1)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
