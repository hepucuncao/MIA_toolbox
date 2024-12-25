import torch
import os 
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler
from PIL import Image
from torch.utils.data import Subset, DataLoader, ConcatDataset
import torch.utils.data as data
from torch._utils import _accumulate
from torch import randperm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def dataset_split(dataset, lengths):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    
    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

    return all_data

class SUBMNIST(Dataset):
    def __init__(self, mode, aug, train):
        self.img_size = 28  # MNIST 图像大小为 28x28
        self.num_classes = 10
        self.mean = [0.1307, 0.1307, 0.1307]  # 修改为三通道的均值
        self.std = [0.3081, 0.3081, 0.3081]  # 修改为三通道的标准差
        
        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # 数据增强
        self.augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.Grayscale(num_output_channels=3),  # 转换为三通道
            transforms.ToTensor(),
            normalize
        ])

        self.normalized = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 转换为三通道
            transforms.ToTensor(),
            normalize
        ])

        # 加载 MNIST 数据集
        self.aug_trainset = datasets.MNIST(root='./c01yili/datasets/MNIST', train=True, download=True, transform=self.augmented)
        self.aug_testset = datasets.MNIST(root='./c01yili/datasets/MNIST', train=False, download=True, transform=self.augmented)
        self.trainset = datasets.MNIST(root='./c01yili/datasets/MNIST', train=True, download=True, transform=self.normalized)
        self.testset = datasets.MNIST(root='./c01yili/datasets/MNIST', train=False, download=True, transform=self.normalized)

        # 连接数据集
        self.aug_dataset = ConcatDataset([self.aug_trainset, self.aug_testset])
        self.dataset = ConcatDataset([self.trainset, self.testset])

        # 数据集分割
        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset, self.aug_distill_testset = dataset_split(self.aug_dataset, [10000, 10000, 10000, 10000, 20000, 10000])
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset, self.distill_testset = dataset_split(self.dataset, [10000, 10000, 10000, 10000, 20000, 10000])

        if mode == 'target':
            if aug:
                self.dataset = self.aug_target_trainset if train else self.aug_target_testset
            else:
                self.dataset = self.target_trainset if train else self.target_testset
        elif mode == 'shadow':
            if aug:
                self.dataset = self.aug_shadow_trainset if train else self.aug_shadow_testset
            else:
                self.dataset = self.shadow_trainset if train else self.shadow_testset
        elif 'distill' in mode:
            if aug:
                self.dataset = self.aug_distill_trainset if train else self.aug_distill_testset
            else:
                self.dataset = self.distill_trainset if train else self.distill_testset

        self.index = range(int(len(self.dataset)))

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)

class MNIST:
    def __init__(self, mode, aug, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 28
        self.num_classes = 10
        self.port_num = 3
        
        if aug:
            if mode == 'target':
                self.aug_target_trainset = SUBMNIST(mode, aug, True)
                self.aug_target_train_loader = DataLoader(self.aug_target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_target_testset = SUBMNIST(mode, aug, False)
                self.aug_target_test_loader = DataLoader(self.aug_target_testset, batch_size=batch_size, shuffle=False, num_workers=2)
            elif mode == 'shadow':
                self.aug_shadow_trainset = SUBMNIST(mode, aug, True)
                self.aug_shadow_train_loader = DataLoader(self.aug_shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_shadow_testset = SUBMNIST(mode, aug, False)
                self.aug_shadow_test_loader = DataLoader(self.aug_shadow_testset, batch_size=batch_size, shuffle=False, num_workers=2)
            elif 'distill' in mode:
                self.aug_distill_trainset = SUBMNIST(mode, aug, True)
                self.aug_distill_train_loader = DataLoader(self.aug_distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_distill_testset = SUBMNIST(mode, aug, False)
                self.aug_distill_test_loader = DataLoader(self.aug_distill_testset, batch_size=batch_size, shuffle=False, num_workers=2)

        else:
            if mode == 'target':
                self.target_trainset = SUBMNIST(mode, aug, True)
                self.target_train_loader = DataLoader(self.target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.target_testset = SUBMNIST(mode, aug, False)
                self.target_test_loader = DataLoader(self.target_testset, batch_size=batch_size, shuffle=False, num_workers=2)
            elif mode == 'shadow':
                self.shadow_trainset = SUBMNIST(mode, aug, True)
                self.shadow_train_loader = DataLoader(self.shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.shadow_testset = SUBMNIST(mode, aug, False)
                self.shadow_test_loader = DataLoader(self.shadow_testset, batch_size=batch_size, shuffle=False, num_workers=2)
            elif 'distill' in mode:
                self.distill_trainset = SUBMNIST(mode, aug, True)
                self.distill_train_loader = DataLoader(self.distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.distill_testset = SUBMNIST(mode, aug, False)
                self.distill_test_loader = DataLoader(self.distill_testset, batch_size=batch_size, shuffle=False, num_workers=2)

                
class GTSRB_ORI(data.Dataset):
    base_folder = 'GTSRB'
    def __init__(self, root_dir, train=False, transform=None):

        self.root_dir = root_dir
        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 0])
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId

class SUBGTSRB(data.Dataset):
    def __init__(self, mode, aug, train):
        self.img_size = 32
        self.num_classes = 43
        self.mean = [0.3403, 0.3121, 0.3214]
        self.std = [0.2724, 0.2608, 0.2669]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])    

        self.aug_trainset = GTSRB_ORI(root_dir='./c01yili/datasets/GTSRB', train=True, transform=self.augmented)
        self.aug_testset = GTSRB_ORI(root_dir='./c01yili/datasets/GTSRB', train=False, transform=self.augmented)
        self.trainset = GTSRB_ORI(root_dir='./c01yili/datasets/GTSRB', train=True, transform=self.normalized)
        self.testset = GTSRB_ORI(root_dir='./c01yili/datasets/GTSRB', train=False, transform=self.normalized)
               
        self.aug_dataset = ConcatDataset([self.aug_trainset, self.aug_testset])
        self.dataset = ConcatDataset([self.trainset, self.testset])

        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset = dataset_split(self.aug_dataset, [1500, 1500, 1500, 1500, 45838])
        self.aug_distill_testset = self.aug_shadow_testset
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset = dataset_split(self.dataset, [1500, 1500, 1500, 1500, 45838])
        self.distill_testset = self.shadow_testset

        if mode == 'target':
            if aug:
                if train:
                    self.dataset = self.aug_target_trainset
                else:
                    self.dataset = self.aug_target_testset
            else:
                if train:
                    self.dataset = self.target_trainset
                else:
                    self.dataset = self.target_testset
        elif mode == 'shadow':
            if aug:
                if train:
                    self.dataset = self.aug_shadow_trainset
                else:
                    self.dataset = self.aug_shadow_testset
            else:
                if train:
                    self.dataset = self.shadow_trainset
                else:
                    self.dataset = self.shadow_testset
        elif 'distill' in mode:
            if aug:
                if train:
                    self.dataset = self.aug_distill_trainset
                else:
                    self.dataset = self.aug_distill_testset
            else:
                if train:
                    self.dataset = self.distill_trainset
                else:
                    self.dataset = self.distill_testset

        self.index = range(int(len(self.dataset)))


    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)
        
class GTSRB:
    def __init__(self, mode, aug, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 43
        self.port_num = 3
        
        if aug:
            if mode == 'target':
                self.aug_target_trainset = SUBGTSRB(mode, aug, True)
                self.aug_target_train_loader = torch.utils.data.DataLoader(self.aug_target_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_target_testset = SUBGTSRB(mode, aug, False)
                self.aug_target_test_loader = torch.utils.data.DataLoader(self.aug_target_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif mode == 'shadow':
                self.aug_shadow_trainset = SUBGTSRB(mode, aug, True)
                self.aug_shadow_train_loader = torch.utils.data.DataLoader(self.aug_shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_shadow_testset = SUBGTSRB(mode, aug, False)
                self.aug_shadow_test_loader = torch.utils.data.DataLoader(self.aug_shadow_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif 'distill' in mode:
                self.aug_distill_trainset = SUBGTSRB(mode, aug, True)
                self.aug_distill_train_loader = torch.utils.data.DataLoader(self.aug_distill_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_distill_testset = SUBGTSRB(mode, aug, False)
                self.aug_distill_test_loader = torch.utils.data.DataLoader(self.aug_distill_testset, batch_size=batch_size, shuffle=True, num_workers=1)

        else:
            if mode == 'target':
                self.target_trainset = SUBGTSRB(mode, aug, True)
                self.target_train_loader = torch.utils.data.DataLoader(self.target_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.target_testset = SUBGTSRB(mode, aug, False)
                self.target_test_loader = torch.utils.data.DataLoader(self.target_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif mode == 'shadow':
                self.shadow_trainset = SUBGTSRB(mode, aug, True)
                self.shadow_train_loader = torch.utils.data.DataLoader(self.shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.shadow_testset = SUBGTSRB(mode, aug, False)
                self.shadow_test_loader = torch.utils.data.DataLoader(self.shadow_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif 'distill' in mode:
                self.distill_trainset = SUBGTSRB(mode, aug, True)
                self.distill_train_loader = torch.utils.data.DataLoader(self.distill_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.distill_testset = SUBGTSRB(mode, aug, False)
                self.distill_test_loader = torch.utils.data.DataLoader(self.distill_testset, batch_size=batch_size, shuffle=True, num_workers=1)

class SUBCINIC10(data.Dataset):
    def __init__(self, mode, aug, train):
        self.img_size = 32
        self.num_classes = 10
        self.mean = [0.47889522, 0.47227842, 0.43047404]
        self.std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/train', transform=self.augmented)
        self.aug_testset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/test', transform=self.augmented)
        self.aug_validset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/valid', transform=self.augmented)
        self.trainset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/train', transform=self.normalized)
        self.testset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/test', transform=self.normalized)
        self.validset =  datasets.ImageFolder(root='./c01yili/datasets/cinic/valid', transform=self.normalized)

        self.aug_dataset = ConcatDataset([self.aug_trainset, self.aug_testset, self.aug_validset])
        self.dataset = ConcatDataset([self.trainset, self.testset, self.validset])
        
        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset, self.aug_distill_testset = dataset_split(self.aug_dataset, [10000, 10000, 10000, 10000, 220000, 10000])
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset, self.distill_testset = dataset_split(self.dataset, [10000, 10000, 10000, 10000, 220000, 10000])

        if mode == 'target':
            if aug:
                if train:
                    self.dataset = self.aug_target_trainset
                else:
                    self.dataset = self.aug_target_testset
            else:
                if train:
                    self.dataset = self.target_trainset
                else:
                    self.dataset = self.target_testset
        elif mode == 'shadow':
            if aug:
                if train:
                    self.dataset = self.aug_shadow_trainset
                else:
                    self.dataset = self.aug_shadow_testset
            else:
                if train:
                    self.dataset = self.shadow_trainset
                else:
                    self.dataset = self.shadow_testset
        elif 'distill' in mode:
            if aug:
                if train:
                    self.dataset = self.aug_distill_trainset
                else:
                    self.dataset = self.aug_distill_testset
            else:
                if train:
                    self.dataset = self.distill_trainset
                else:
                    self.dataset = self.distill_testset

        self.index = range(int(len(self.dataset)))


    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)

class CINIC10:
    def __init__(self, mode, aug, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.port_num = 3
        
        if aug:
            if mode == 'target':
                self.aug_target_trainset = SUBCINIC10(mode, aug, True)
                self.aug_target_train_loader = torch.utils.data.DataLoader(self.aug_target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_target_testset = SUBCINIC10(mode, aug, False)
                self.aug_target_test_loader = torch.utils.data.DataLoader(self.aug_target_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif mode == 'shadow':
                self.aug_shadow_trainset = SUBCINIC10(mode, aug, True)
                self.aug_shadow_train_loader = torch.utils.data.DataLoader(self.aug_shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_shadow_testset = SUBCINIC10(mode, aug, False)
                self.aug_shadow_test_loader = torch.utils.data.DataLoader(self.aug_shadow_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif 'distill' in mode:
                self.aug_distill_trainset = SUBCINIC10(mode, aug, True)
                self.aug_distill_train_loader = torch.utils.data.DataLoader(self.aug_distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_distill_testset = SUBCINIC10(mode, aug, False)
                self.aug_distill_test_loader = torch.utils.data.DataLoader(self.aug_distill_testset, batch_size=batch_size, shuffle=True, num_workers=2)

        else:
            if mode == 'target':
                self.target_trainset = SUBCINIC10(mode, aug, True)
                self.target_train_loader = torch.utils.data.DataLoader(self.target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.target_testset = SUBCINIC10(mode, aug, False)
                self.target_test_loader = torch.utils.data.DataLoader(self.target_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif mode == 'shadow':
                self.shadow_trainset = SUBCINIC10(mode, aug, True)
                self.shadow_train_loader = torch.utils.data.DataLoader(self.shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.shadow_testset = SUBCINIC10(mode, aug, False)
                self.shadow_test_loader = torch.utils.data.DataLoader(self.shadow_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif 'distill' in mode:
                self.distill_trainset = SUBCINIC10(mode, aug, True)
                self.distill_train_loader = torch.utils.data.DataLoader(self.distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.distill_testset = SUBCINIC10(mode, aug, False)
                self.distill_test_loader = torch.utils.data.DataLoader(self.distill_testset, batch_size=batch_size, shuffle=True, num_workers=2)

class SUBCIFAR10(data.Dataset):
    def __init__(self, mode, aug, train):
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR10(root='./c01yili/datasets/CIFAR10', train=True, download=True, transform=self.augmented)
        self.aug_testset =  datasets.CIFAR10(root='./c01yili/datasets/CIFAR10', train=False, download=True, transform=self.augmented)
        self.trainset =  datasets.CIFAR10(root='./c01yili/datasets/CIFAR10', train=True, download=True, transform=self.normalized)
        self.testset =  datasets.CIFAR10(root='./c01yili/datasets/CIFAR10', train=False, download=True, transform=self.normalized)

        self.aug_dataset = ConcatDataset([self.aug_trainset, self.aug_testset])
        self.dataset = ConcatDataset([self.trainset, self.testset])
        
        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset = dataset_split(self.aug_dataset, [10000, 10000, 10000, 10000, 20000])
        self.aug_distill_testset = self.aug_shadow_testset
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset = dataset_split(self.dataset, [10000, 10000, 10000, 10000, 20000])
        self.distill_testset = self.shadow_testset

        if mode == 'target':
            if aug:
                if train:
                    self.dataset = self.aug_target_trainset
                else:
                    self.dataset = self.aug_target_testset
            else:
                if train:
                    self.dataset = self.target_trainset
                else:
                    self.dataset = self.target_testset
        elif mode == 'shadow':
            if aug:
                if train:
                    self.dataset = self.aug_shadow_trainset
                else:
                    self.dataset = self.aug_shadow_testset
            else:
                if train:
                    self.dataset = self.shadow_trainset
                else:
                    self.dataset = self.shadow_testset
        elif 'distill' in mode:
            if aug:
                if train:
                    self.dataset = self.aug_distill_trainset
                else:
                    self.dataset = self.aug_distill_testset
            else:
                if train:
                    self.dataset = self.distill_trainset
                else:
                    self.dataset = self.distill_testset

        self.index = range(int(len(self.dataset)))


    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)

class CIFAR10:
    def __init__(self, mode, aug, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.port_num = 3
        
        if aug:
            if mode == 'target':
                self.aug_target_trainset = SUBCIFAR10(mode, aug, True)
                self.aug_target_train_loader = torch.utils.data.DataLoader(self.aug_target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_target_testset = SUBCIFAR10(mode, aug, False)
                self.aug_target_test_loader = torch.utils.data.DataLoader(self.aug_target_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif mode == 'shadow':
                self.aug_shadow_trainset = SUBCIFAR10(mode, aug, True)
                self.aug_shadow_train_loader = torch.utils.data.DataLoader(self.aug_shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_shadow_testset = SUBCIFAR10(mode, aug, False)
                self.aug_shadow_test_loader = torch.utils.data.DataLoader(self.aug_shadow_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif 'distill' in mode:
                self.aug_distill_trainset = SUBCIFAR10(mode, aug, True)
                self.aug_distill_train_loader = torch.utils.data.DataLoader(self.aug_distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.aug_distill_testset = SUBCIFAR10(mode, aug, False)
                self.aug_distill_test_loader = torch.utils.data.DataLoader(self.aug_distill_testset, batch_size=batch_size, shuffle=True, num_workers=2)

        else:
            if mode == 'target':
                self.target_trainset = SUBCIFAR10(mode, aug, True)
                self.target_train_loader = torch.utils.data.DataLoader(self.target_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.target_testset = SUBCIFAR10(mode, aug, False)
                self.target_test_loader = torch.utils.data.DataLoader(self.target_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif mode == 'shadow':
                self.shadow_trainset = SUBCIFAR10(mode, aug, True)
                self.shadow_train_loader = torch.utils.data.DataLoader(self.shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.shadow_testset = SUBCIFAR10(mode, aug, False)
                self.shadow_test_loader = torch.utils.data.DataLoader(self.shadow_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            elif 'distill' in mode:
                self.distill_trainset = SUBCIFAR10(mode, aug, True)
                self.distill_train_loader = torch.utils.data.DataLoader(self.distill_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
                self.distill_testset = SUBCIFAR10(mode, aug, False)
                self.distill_test_loader = torch.utils.data.DataLoader(self.distill_testset, batch_size=batch_size, shuffle=True, num_workers=2)
            
class SUBCIFAR100(data.Dataset):
    def __init__(self, mode, aug, train):
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000
        self.mean=[0.507, 0.487, 0.441]
        self.std=[0.267, 0.256, 0.276]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR100(root='./c01yili/datasets/CIFAR100', train=True, download=True, transform=self.augmented)
        self.aug_testset =  datasets.CIFAR100(root='./c01yili/datasets/CIFAR100', train=False, download=True, transform=self.augmented)
        self.trainset =  datasets.CIFAR100(root='./c01yili/datasets/CIFAR100', train=True, download=True, transform=self.normalized)
        self.testset =  datasets.CIFAR100(root='./c01yili/datasets/CIFAR100', train=False, download=True, transform=self.normalized)

        self.aug_dataset = ConcatDataset([self.aug_trainset, self.aug_testset])
        self.dataset = ConcatDataset([self.trainset, self.testset])
        
        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset = dataset_split(self.aug_dataset, [10000, 10000, 10000, 10000, 20000])
        self.aug_distill_testset = self.aug_shadow_testset
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset = dataset_split(self.dataset, [10000, 10000, 10000, 10000, 20000])
        self.distill_testset = self.shadow_testset

        if mode == 'target':
            if aug:
                if train:
                    self.dataset = self.aug_target_trainset
                else:
                    self.dataset = self.aug_target_testset
            else:
                if train:
                    self.dataset = self.target_trainset
                else:
                    self.dataset = self.target_testset
        elif mode == 'shadow':
            if aug:
                if train:
                    self.dataset = self.aug_shadow_trainset
                else:
                    self.dataset = self.aug_shadow_testset
            else:
                if train:
                    self.dataset = self.shadow_trainset
                else:
                    self.dataset = self.shadow_testset
        elif 'distill' in mode:
            if aug:
                if train:
                    self.dataset = self.aug_distill_trainset
                else:
                    self.dataset = self.aug_distill_testset
            else:
                if train:
                    self.dataset = self.distill_trainset
                else:
                    self.dataset = self.distill_testset

        self.index = range(int(len(self.dataset)))

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)

class CIFAR100:
    def __init__(self, mode, aug, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.port_num = 3
        if aug:
            if mode == 'target':
                self.aug_target_trainset = SUBCIFAR100(mode, aug, True)
                self.aug_target_train_loader = torch.utils.data.DataLoader(self.aug_target_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_target_testset = SUBCIFAR100(mode, aug, False)
                self.aug_target_test_loader = torch.utils.data.DataLoader(self.aug_target_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif mode == 'shadow':
                self.aug_shadow_trainset = SUBCIFAR100(mode, aug, True)
                self.aug_shadow_train_loader = torch.utils.data.DataLoader(self.aug_shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_shadow_testset = SUBCIFAR100(mode, aug, False)
                self.aug_shadow_test_loader = torch.utils.data.DataLoader(self.aug_shadow_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif 'distill' in mode:
                self.aug_distill_trainset = SUBCIFAR100(mode, aug, True)
                self.aug_distill_train_loader = torch.utils.data.DataLoader(self.aug_distill_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.aug_distill_testset = SUBCIFAR100(mode, aug, False)
                self.aug_distill_test_loader = torch.utils.data.DataLoader(self.aug_distill_testset, batch_size=batch_size, shuffle=True, num_workers=1)

        else:
            if mode == 'target':
                self.target_trainset = SUBCIFAR100(mode, aug, True)
                self.target_train_loader = torch.utils.data.DataLoader(self.target_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.target_testset = SUBCIFAR100(mode, aug, False)
                self.target_test_loader = torch.utils.data.DataLoader(self.target_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif mode == 'shadow':
                self.shadow_trainset = SUBCIFAR100(mode, aug, True)
                self.shadow_train_loader = torch.utils.data.DataLoader(self.shadow_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.shadow_testset = SUBCIFAR100(mode, aug, False)
                self.shadow_test_loader = torch.utils.data.DataLoader(self.shadow_testset, batch_size=batch_size, shuffle=True, num_workers=1)
            elif 'distill' in mode:
                self.distill_trainset = SUBCIFAR100(mode, aug, True)
                self.distill_train_loader = torch.utils.data.DataLoader(self.distill_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
                self.distill_testset = SUBCIFAR100(mode, aug, False)
                self.distill_test_loader = torch.utils.data.DataLoader(self.distill_testset, batch_size=batch_size, shuffle=True, num_workers=1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res
