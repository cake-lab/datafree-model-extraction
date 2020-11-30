from torchvision import datasets, transforms
import torch


def get_dataloader(args):
    if args.dataset.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
                            

    elif args.dataset.lower()=='svhn':
        print("Loading SVHN data")
        train_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(args.data_root, split='train', download=True,
                       transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(args.data_root, split='test', download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif args.dataset.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader