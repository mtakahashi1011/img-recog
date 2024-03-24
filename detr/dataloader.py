from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import CocoDetection
from transform import train_transform, valid_transform, collate_func
from retinanet.utils import generate_subset

def create_dataloader(config):
    train_dataset = CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file, transform=train_transform())
    val_dataset = CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file, transform=valid_transform())
    print(len(train_dataset.classes))

    val_set, train_set = generate_subset(train_dataset, config.val_ratio)

    print(f'学習セットのサンプル数: {len(train_set)}')
    print(f'検証セットのサンプル数: {len(val_set)}')

    train_sampler = SubsetRandomSampler(train_set)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=train_sampler,
        collate_fn=collate_func)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, sampler=val_set,
        collate_fn=collate_func)
    return train_loader, val_loader