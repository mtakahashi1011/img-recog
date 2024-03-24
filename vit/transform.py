import numpy as np 
import torchvision.transforms as T


def transform(channel_mean: np.ndarray, channel_std: np.ndarray):
    transforms = [
        T.ToTensor(), # transform PIL Image(W,H,C) into Tensor(C,W,H) of type torch.Float(32bit float) whose values are in the range [0.0,1.0]
        T.Normalize(mean=channel_mean, std=channel_std)
    ]
    return T.Compose(transforms)