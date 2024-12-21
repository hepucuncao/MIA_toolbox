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

    if not os.path.exists("./delect_weights_15"):
        os.makedirs("./delect_weights_15")

    tb_writer = SummaryWriter()

    # 定义数据转换
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

    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root='cifar100', train=True, download=True, transform=data_transform["train"])
    val_dataset = datasets.CIFAR100(root='cifar100', train=False, download=True, transform=data_transform["val"])

    # 定义数据删除函数
    def get_subset_indices(dataset, delete_ratio):
        total_len = len(dataset)
        delete_count = int(total_len * delete_ratio)
        all_indices = list(range(total_len))
        delete_indices = random.sample(all_indices, delete_count)
        remaining_indices = list(set(all_indices) - set(delete_indices))
        return remaining_indices, delete_indices

    # 获取不同数据集的索引
    full_indices = list(range(len(train_dataset)))
    train_indices_5, deleted_indices_5 = get_subset_indices(train_dataset, delete_ratio=0.05)
    train_indices_10, deleted_indices_10 = get_subset_indices(train_dataset, delete_ratio=0.10)

    # 记录删除的样本
    print(f"Deleted 5% data indices: {deleted_indices_5}")
    print(f"Deleted 10% data indices: {deleted_indices_10}")

    # 定义数据加载器函数
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers

    def get_dataloader(train_indices):
        train_subset = Subset(train_dataset, train_indices)
        return DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)

    # 用不同的数据集训练模型
    for subset_name, train_indices in zip(["full", "delete_5", "delete_10"],
                                          [full_indices, train_indices_5, train_indices_10]):
        print(f"\nTraining model with {subset_name} dataset")

        train_loader = get_dataloader(train_indices)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

        model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits else ['pre_logits.fc.weight',
                                                                              'pre_logits.fc.bias', 'head.weight',
                                                                              'head.bias']
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

        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            scheduler.step()

            # validate
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

            # 保存模型权重
            torch.save(model.state_dict(), f"./delect_weights_15/model_{subset_name}_epoch_{epoch}.pth")


if __name__ == '__main__':
    class Args:
        num_classes = 100  # CIFAR-100的类别数
        epochs = 15
        batch_size = 8
        lr = 0.001
        lrf = 0.01
        model_name = ""
        weights = 'vit_base_patch16_224_in21k.pth'
        freeze_layers = True
        device = "cuda:0"


    opt = Args()
    main(opt)
