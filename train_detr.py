import json
import torch
from torch import optim
from tqdm import tqdm
from collections import deque
from pydantic import BaseModel
from pycocotools.cocoeval import COCOeval

from detr.dataloader import create_dataloader
from detr.model import DETR
from detr.loss import _loss_func


URL = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

class ModelConfig(BaseModel):
    num_queries: int = 100                             # 物体クエリ埋め込みのクエリベクトル数
    dim_hidden: int = 256                              # Transformer内の特徴量次元
    num_heads: int = 8                                 # マルチヘッドアテンションのヘッド数
    num_encoder_layers: int = 6                        # Transformerエンコーダの層数
    num_decoder_layers: int = 6                        # Transformerデコーダの層数
    dim_feedforward: int = 2048                        # Transformer内のFNNの中間特徴量次元
    dropout: float = 0.1                               # Transformer内のドロップアウト率
    num_classes: int = 2


class LossConfig(BaseModel):
    loss_weight_class: int = 1                         # 分類損失の重み
    loss_weight_box_l1: int = 5                        # 矩形回帰のL1誤差の重み
    loss_weight_box_giou: int = 2                      # 矩形回帰のGIoU損失の重み
    background_weight: float = 0.1                     # 背景クラスの重み        


class ConfigTrainEval:
    def __init__(self):
        self.img_directory = 'data/val2014/'                     # 画像があるディレクトリ
        self.anno_file = 'data/instances_val2014_small.json' # アノテーションファイルのパス
        self.save_file = 'cocodataset/detr.pth'                     # パラメータを保存するパス
        self.val_ratio = 0.2                               # 検証に使う学習セット内のデータの割合
        self.num_epochs = 100                              # 学習エポック数
        self.lr_drop = 90                                  # 学習率を減衰させるエポック
        self.val_interval = 5                              # 検証を行うエポック間隔
        self.lr = 1e-4                                     # 学習率
        self.lr_backbone = 1e-5                            # バックボーンネットワークの学習率
        self.weight_decay = 1e-4                           # 荷重減衰
        self.clip = 0.1                                    # 勾配のクリップ上限

        self.moving_avg = 100                              # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 1                                # バッチサイズ
        self.num_workers = 2                               # データローダに使うCPUプロセスの数
        self.device = 'mps'                                # 学習に使うデバイス


def evaluate(data_loader, model, loss_func):
    model.eval()
    losses_class = []
    losses_box_l1 = []
    losses_box_giou = []
    losses_aux = []
    losses = []
    preds = []
    img_ids = []
    for imgs, masks, targets in tqdm(data_loader, desc='[Validation]'):
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            masks = masks.to(model.get_device())
            targets = [{k: v.to(model.get_device()) for k, v in target.items()} for target in targets]

            preds_class, preds_box = model(imgs, masks)
            num_decoder_layers = preds_class.shape[0]

            loss_aux = 0
            for layer_index in range(num_decoder_layers-1):
                loss_aux += sum(loss_func(preds_class[layer_index], preds_box[layer_index], targets))
            
            loss_class, loss_box_l1, loss_box_giou = loss_func(preds_class[-1], preds_box[-1], targets)
            loss = loss_class + loss_box_l1 + loss_box_giou + loss_aux

            losses_class.append(loss_class)
            losses_box_l1.append(loss_box_l1)
            losses_box_giou.append(loss_box_giou)
            losses_aux.append(loss_aux)
            losses.append(loss)
    loss_class = torch.stack(losses_class).mean().item()
    loss_box_l1 = torch.stack(losses_box_l1).mean().item()
    loss_box_giou = torch.stack(losses_box_giou).mean().item()
    loss_aux = torch.stack(losses_aux).mean().item()
    loss = torch.stack(losses).mean().item()
    print(f'Validation loss = {loss:.3f}, '
          f'class loss = {loss_class:.3f}, '
          f'box l1 loss = {loss_box_l1:.3f}, '
          f'box giou loss = {loss_box_giou:.3f}, '
          f'aux loss = {loss_aux:.3f}')

    if len(preds) == 0:
        print('Nothing detected, skip evaluation.')
        return

    with open('tmp.json', 'w') as f:
        json.dump(preds, f)

    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

    coco_eval = COCOeval(data_loader.dataset.coco, coco_results, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()                        


def train_eval():
    config = ConfigTrainEval()
    train_loader, val_loader = create_dataloader(config)
    
    model_config = ModelConfig()
    model = DETR(**model_config.model_dump())
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(URL), strict=False)
    model.to(config.device)    

    params_backbone = []
    params_others = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if 'backbone' in name:
                params_backbone.append(parameter)
            else:
                params_others.append(parameter)
    param_groups = [
        {'params': params_backbone, 'lr': config.lr_backbone},
        {'params': params_others, 'lr': config.lr}]
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    # 指定したエポックで学習率を1/10に減衰するスケジューラを生成
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.lr_drop], gamma=0.1)

    loss_config = LossConfig()
    loss_func = lambda preds_class, preds_box, targets: _loss_func(preds_class, preds_box, targets, **loss_config.model_dump())

    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')
            losses_class = deque()
            losses_box_l1 = deque()
            losses_box_giou = deque()
            losses_aux = deque()
            losses = deque()
            for imgs, masks, targets in pbar:
                imgs = imgs.to(model.get_device())
                masks = masks.to(model.get_device())
                targets = [{k: v.to(model.get_device()) for k, v in target.items()} for target in targets]

                optimizer.zero_grad()

                preds_class, preds_box = model(imgs, masks)

                loss_aux = 0
                for layer_index in range(model_config.num_decoder_layers - 1):
                    loss_aux += sum(loss_func(preds_class[layer_index], preds_box[layer_index], targets))
                loss_class, loss_box_l1, loss_box_giou = loss_func(preds_class[-1], preds_box[-1], targets)
                loss = loss_aux + loss_class + loss_box_l1 + loss_box_giou

                loss.backward()

                # 勾配全体のL2ノルムが上限を越えるとき上限値でクリップ
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)

                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box_l1.append(loss_box_l1.item())
                losses_box_giou.append(loss_box_giou.item())
                losses_aux.append(loss_aux.item())
                losses.append(loss.item())
                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box_l1.popleft()
                    losses_box_giou.popleft()
                    losses_aux.popleft()
                    losses.popleft()
                pbar.set_postfix(
                    {'loss': torch.Tensor(losses).mean().item(),
                     'loss_class': torch.Tensor(losses_class).mean().item(),
                     'loss_box_l1': torch.Tensor(losses_box_l1).mean().item(),
                     'loss_box_giou': torch.Tensor(losses_box_giou).mean().item(),
                     'loss_aux': torch.Tensor(losses_aux).mean().item()})
                
        scheduler.step()
        torch.save(model.state_dict(), config.save_file)
        if (epoch + 1) % config.val_interval == 0:
            evaluate(val_loader, model, loss_func)

if __name__ == "__main__":
    train_eval()