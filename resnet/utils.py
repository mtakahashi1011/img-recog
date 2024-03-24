import random
import torch
from torch.utils.data import Dataset


def generate_subset(dataset: Dataset, ratio: float, random_seed: int=0):
    size = int(len(dataset) * ratio)
    indices = list(range(len(dataset)))
    random.seed(random_seed)
    random.shuffle(indices)
    indices1, indices2 = indices[:size], indices[size:]
    return indices1, indices2 

def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        img = dataset[i][0]
        data.append(img)
    data = torch.stack(data) # (N, C, W, H)
    channel_mean = data.mean(dim=(0, 2, 3)) # dim: the dimension or dimensions to reduce
    channel_std = data.std(dim=(0, 2, 3))
    return channel_mean, channel_std