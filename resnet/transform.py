import numpy as np 
import torchvision.transforms as T


def transform(channel_mean: np.ndarray, channel_std: np.ndarray):
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std)
    ]
    return T.Compose(transforms)