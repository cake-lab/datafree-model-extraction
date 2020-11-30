from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import ipdb
import torch

class RandDataset(Dataset):
    #Used for sampling Random queries for model Extraction Attacks on SVHN
    def __init__(self, batch_size, max_examples):
        self.batch_size = batch_size
        self.channels = 3
        self.pix = 32
        self.len = int(max_examples/batch_size)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        z = torch.randn((self.batch_size, 3, 32, 32))
        labels = torch.randn((self.batch_size,))
        sample = [z,labels]

        return sample

    

def data_loader(dataset, batch_size, max_examples):
    if dataset == "random":
        train_loader = RandDataset(batch_size = batch_size, max_examples = max_examples)
        return train_loader, train_loader

    func = {"svhn":datasets.SVHN, "svhn_skew":datasets.SVHN, "cifar10":datasets.CIFAR10, "cifar100":datasets.CIFAR100, "mnist":datasets.MNIST, "Imagenet":datasets.ImageNet,"MNIST-M":None}
    norm_mean = {"svhn":(0.438, 0.444, 0.473), "svhn_skew":(0.438, 0.444, 0.473), "cifar10":(0.4914, 0.4822, 0.4465), "cifar100":(0.4914, 0.4822, 0.4465), "mnist":(0.1307,), "Imagenet":(0.485, 0.456, 0.406),"MNIST-M":None}
    norm_std = {"svhn":(0.198, 0.201, 0.197), "svhn_skew":(0.198, 0.201, 0.197), "cifar10":(0.2023, 0.1994, 0.2010), "cifar100":(0.2023, 0.1994, 0.2010), "mnist": (0.3081,), "Imagenet":(0.229, 0.224, 0.225),"MNIST-M":None}


    tr_normalize = transforms.Normalize(norm_mean[dataset], norm_std[dataset])
    transform_train = transforms.Compose([
                                    transforms.Resize((32, 32), interpolation=PIL.Image.BILINEAR),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), tr_normalize,
                                    transforms.Lambda(lambda x: x.float())])
    transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])

    data_source = func[dataset]
    try:
        d_train = data_source("../data", train=True, download=True, transform=transform_train)
        d_test = data_source("../data", train=False, download=True, transform=transform_test)
    except:
        d_train = data_source("../data", split='train' if dataset != "svhn_skew" else 'extra', download=True, transform=transform_train)
        d_test = data_source("../data", split='test', download=True, transform=transform_test)

    if dataset == 'svhn_skew':
        d_train.data = d_train.data[d_train.labels < 5]
        d_train.labels = d_train.labels[d_train.labels < 5]
    if dataset in ['svhn', 'svhn_skew']:
        ## Cap maximum size to 50_000 unique samples
        d_train.data = d_train.data[:50000]
        d_train.labels = d_train.labels[:50000]
    if dataset in ['mnist']:
        d_train.data = d_train.data[:50000]
        d_train.targets = d_train.targets[:50000]

    train_loader = DataLoader(d_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(d_test, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader

