import os
import math
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

from net import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./part_weights_15"):
        os.makedirs("./part_weights_15")

    tb_writer = SummaryWriter()

    # 数据转换
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # 加载 CIFAR-100 数据集
    train_dataset = datasets.CIFAR100(root='cifar100', train=True, download=True, transform=data_transform["train"])
    val_dataset = datasets.CIFAR100(root='cifar100', train=False, download=True, transform=data_transform["val"])

    # 获取删除数据集的索引
    def get_subset_indices(dataset, delete_ratio):
        total_len = len(dataset)
        delete_count = int(total_len * delete_ratio)
        all_indices = list(range(total_len))
        delete_indices = random.sample(all_indices, delete_count)
        remaining_indices = list(set(all_indices) - set(delete_indices))
        return remaining_indices, delete_indices

    # 设置完整数据、5%和10%删除数据的索引
    full_indices = list(range(len(train_dataset)))
    train_indices_5, deleted_indices_5 = get_subset_indices(train_dataset, delete_ratio=0.05)
    train_indices_10, deleted_indices_10 = get_subset_indices(train_dataset, delete_ratio=0.10)

    # 记录删除的数据
    print(f"Deleted 5% data indices: {deleted_indices_5}")
    print(f"Deleted 10% data indices: {deleted_indices_10}")

    # 定义数据加载器
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    def get_dataloader(train_indices):
        train_subset = Subset(train_dataset, train_indices)
        return DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)

    # 完整训练并保存模型
    print("Training model with full dataset")
    train_loader = get_dataloader(full_indices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 完整训练循环
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(f"{tags[0]}_full", train_loss, epoch)
        tb_writer.add_scalar(f"{tags[1]}_full", train_acc, epoch)
        tb_writer.add_scalar(f"{tags[2]}_full", val_loss, epoch)
        tb_writer.add_scalar(f"{tags[3]}_full", val_acc, epoch)
        tb_writer.add_scalar(f"{tags[4]}_full", optimizer.param_groups[0]["lr"], epoch)
    torch.save(model.state_dict(), "./part_weights_15/model_full.pth")

    # 分组重训练
    for subset_name, delete_indices in zip(["delete_5", "delete_10"], [deleted_indices_5, deleted_indices_10]):
        print(f"\nFine-tuning model with {subset_name} dataset")
        finetune_loader = get_dataloader(delete_indices)

        model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
        model.load_state_dict(torch.load("./part_weights_15/model_full.pth"))

        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, momentum=0.9, weight_decay=5E-5)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # 分组重训练循环
        for epoch in range(args.finetune_epochs):
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=finetune_loader,
                                                    device=device,
                                                    epoch=epoch)
            scheduler.step()
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(f"{tags[0]}_{subset_name}", train_loss, epoch)
            tb_writer.add_scalar(f"{tags[1]}_{subset_name}", train_acc, epoch)
            tb_writer.add_scalar(f"{tags[2]}_{subset_name}", val_loss, epoch)
            tb_writer.add_scalar(f"{tags[3]}_{subset_name}", val_acc, epoch)
            tb_writer.add_scalar(f"{tags[4]}_{subset_name}", optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), f"./part_weights_15/model_{subset_name}.pth")


if __name__ == '__main__':
    class Args:
        num_classes = 100  # CIFAR-100的类别数
        epochs = 15
        finetune_epochs = 5  # 分组重训练的epoch数
        batch_size = 8
        lr = 0.001
        lrf = 0.01
        model_name = ""
        weights = 'vit_base_patch16_224_in21k.pth'
        freeze_layers = True
        device = "cuda:0"

    opt = Args()
    main(opt)
