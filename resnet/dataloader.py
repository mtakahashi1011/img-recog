from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision 
import torchvision.transforms as T

from resnet.utils import get_dataset_statistics, generate_subset
from resnet.transform import transform


def create_dataloader(config):
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=T.ToTensor()
    )
    channel_mean, channel_std = get_dataset_statistics(dataset)
    transforms = transform(channel_mean, channel_std)

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms
    )
    val_set, train_set = generate_subset(train_dataset, config.val_ratio)
    train_sampler = SubsetRandomSampler(train_set)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=train_sampler
    )
    valid_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=val_set
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )
    return train_loader, valid_loader, test_loader