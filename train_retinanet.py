from tqdm import tqdm
from collections import deque
from typing import Callable
import json
from pycocotools.cocoeval import COCOeval

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from retinanet.dataloader import create_dataloader
from retinanet.model import RetinaNet
from retinanet.post_process import post_process
from retinanet.loss import loss_func


class ConfigTrainEval:
    def __init__(self):
        self.img_directory = 'data/val2014'                     # 画像があるディレクトリ
        self.anno_file = 'data/instances_val2014_small.json' # アノテーションファイルのパス
        self.save_file = 'cocodataset/retinanet.pth'                # パラメータを保存するパス
        self.val_ratio = 0.2                               # 検証に使う学習セット内のデータの割合
        self.num_epochs = 50                               # 学習エポック数
        self.lr_drop = 45                                  # 学習率を減衰させるエポック
        self.val_interval = 5                              # 検証を行うエポック間隔
        self.lr = 1e-5                                     # 学習率
        self.clip = 0.1                                    # 勾配のクリップ上限
        self.moving_avg = 100                              # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 8                                # バッチサイズ
        self.num_workers = 4                               # データローダに使うCPUプロセスの数
        self.device = 'mps'


def evaluate(data_loader: DataLoader, model: nn.Module, loss_func: Callable, conf_threshold: float=0.05, nms_threshold: float=0.5):
    model.eval()

    losses_class = []
    losses_box = []
    losses = []
    preds = []
    img_ids = []
    for imgs, targets in tqdm(data_loader, desc='[Validation]'):
        with  torch.no_grad():
            imgs = imgs.to(model.get_device())
            targets = [{k: v.to(model.get_device())
                        for k, v in target.items()}
                       for target in targets]

            preds_class, preds_box, anchors = model(imgs)

            loss_class, loss_box = loss_func(
                preds_class, preds_box, anchors, targets)
            loss = loss_class + loss_box

            losses_class.append(loss_class)
            losses_box.append(loss_box)
            losses.append(loss)

            # 後処理により最終的な検出矩形を取得
            scores, labels, boxes = post_process(
                preds_class, preds_box, anchors, targets,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold)

            for img_scores, img_labels, img_boxes, img_targets in zip(
                    scores, labels, boxes, targets):
                img_ids.append(img_targets['image_id'].item())

                # 評価のためにCOCOの元々の矩形表現である
                # xmin, ymin, width, heightに変換
                img_boxes[:, 2:] -= img_boxes[:, :2]

                for score, label, box in zip(
                        img_scores, img_labels, img_boxes):
                    # COCO評価用のデータの保存
                    preds.append({
                        'image_id': img_targets['image_id'].item(),
                        'category_id': \
                        data_loader.dataset.to_coco_label(
                            label.item()),
                        'score': score.item(),
                        'bbox': box.to('cpu').numpy().tolist()
                    })

    loss_class = torch.stack(losses_class).mean().item()
    loss_box = torch.stack(losses_box).mean().item()
    loss = torch.stack(losses).mean().item()
    print(f'Validation loss = {loss:.3f},'
          f'class loss = {loss_class:.3f}, '
          f'box loss = {loss_box:.3f} ')

    if len(preds) == 0:
        print('Nothing detected, skip evaluation.')

        return

    # COCOevalクラスを使って評価するには検出結果を
    # jsonファイルに出力する必要があるため、jsonファイルに一時保存
    with open('tmp.json', 'w') as f:
        json.dump(preds, f)

    # 一時保存した検出結果をCOCOクラスを使って読み込み
    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

    # COCOevalクラスを使って評価
    coco_eval = COCOeval(
        data_loader.dataset.coco, coco_results, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

init_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def train_eval():
    config = ConfigTrainEval()
    train_loader, val_loader = create_dataloader(config)

    model = RetinaNet(2)
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(init_url), strict=False)
    model.to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.lr_drop], gamma=0.1)

    for epoch in range(config.num_epochs):
        model.train()
        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')
            losses_class = deque()
            losses_box = deque()
            losses = deque()
            for imgs, targets in pbar:
                imgs = imgs.to(model.get_device())
                targets = [{k: v.to(model.get_device()) for k, v in target.items()} for target in targets]
                optimizer.zero_grad()

                preds_class, preds_box, anchors = model(imgs)
                loss_class, loss_box = loss_func(preds_class, preds_box, anchors, targets)
                loss = loss_class + loss_box
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box.append(loss_box.item())
                losses.append(loss.item())
                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box.popleft()
                    losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'loss_class': torch.Tensor(losses_class).mean().item(),
                    'loss_box': torch.Tensor(losses_box).mean().item()})
        scheduler.step()
        torch.save(model.state_dict(), config.save_file)
        if (epoch + 1) % config.val_interval == 0:
            evaluate(val_loader, model, loss_func)

if __name__ == "__main__":
    train_eval()