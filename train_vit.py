import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import deque
from tqdm import tqdm
from typing import Callable

from vit.dataloader import create_dataloader
from vit.model import VisionTransformer


class Config:
    def __init__(self):
        self.val_ratio = 0.2
        self.patch_size = 4
        self.dim_hidden = 512
        self.num_heads = 8
        self.dim_feedforward = 512
        self.num_layers = 6
        self.num_epochs = 30
        self.lr = 1e-2
        self.moving_avg = 20
        self.batch_size = 32
        self.num_workers = 2
        self.device = "mps"
        self.num_samples = 200
        self.num_classes = 10

def evaluate(data_loader: DataLoader, model: nn.Module, loss_func: Callable):
    model.eval()
    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())
            y_pred = model(x)
            losses.append(loss_func(y_pred, y, reduction='none'))
            preds.append(y_pred.argmax(dim=1) == y)
    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()
    return loss, accuracy

def train_eval():
    config = Config()
    train_loader, val_loader, test_loader = create_dataloader(config)
    loss_func = F.cross_entropy

    val_loss_best = float('inf')
    model_best = None 

    model = VisionTransformer(config.num_classes, 32, config.patch_size, config.dim_hidden, config.num_heads, 
                              config.dim_feedforward, config.num_layers)
    model.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch+1}]')
            losses = deque()
            accs = deque()
            for x, y in pbar:
                x = x.to(model.get_device())
                y = y.to(model.get_device())
                optimizer.zero_grad()

                y_pred = model(x)
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'accuracy': torch.Tensor(accs).mean().item()
                })
        val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
        print(f'検証 : loss = {val_loss:.3f} , 'f'accuracy = {val_accuracy:.3f}')
        if val_loss < val_loss_best:
            val_loss_best = val_loss 
            model_best = model.copy()

    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f'テスト : loss = {test_loss:.3f}, 'f'accuracy = {test_accuracy:.3f}')
    
if __name__ == "__main__":
    train_eval()