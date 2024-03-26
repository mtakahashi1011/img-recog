from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from detr.dataset import CocoDetection
from detr.transform import train_transform, valid_transform, collate_func
from detr.utils import generate_subset


def create_dataloader(config):
    train_dataset = CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file,
        transform=train_transform()
    )
    val_dataset = CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file,
        transform=valid_transform()
    )

    val_set, train_set = generate_subset(train_dataset, config.val_ratio)
    train_sampler = SubsetRandomSampler(train_set)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        collate_fn=collate_func
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_set,
        collate_fn=collate_func
    )
    return train_loader, val_loader