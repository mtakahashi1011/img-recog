import numpy as np
from typing import Callable

import torch
import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_directory: str, anno_file: str,
                 transform: Callable=None):
        super().__init__(img_directory, anno_file)

        self.transform = transform

        self.classes = []
        self.coco_to_pred = {}
        self.pred_to_coco = {}
        for i, category_id in enumerate(
                sorted(self.coco.cats.keys())):
            self.classes.append(self.coco.cats[category_id]['name'])
            self.coco_to_pred[category_id] = i
            self.pred_to_coco[i] = category_id

    def __getitem__(self, idx: int):
        img, target = super().__getitem__(idx)
        img_id = self.ids[idx]

        # 物体の集合を一つの矩形でアノテーションしているものを除外
        target = [obj for obj in target if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        classes = torch.tensor([self.coco_to_pred[obj['category_id']] for obj in target], dtype=torch.int64)
        boxes = torch.tensor([obj['bbox'] for obj in target], dtype=torch.float32)

        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4))

        width, height = img.size
        # xmin, ymin, width, height -> xmin, ymin, xmax, ymax
        boxes[:, 2:] += boxes[:, :2]

        boxes[:, ::2] = boxes[:, ::2].clamp(min=0, max=width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)

        target = {
            'image_id': torch.tensor(img_id, dtype=torch.int64),
            'classes': classes,
            'boxes': boxes,
            'size': torch.tensor((width, height), dtype=torch.int64),
            'orig_size': torch.tensor((width, height), dtype=torch.int64),
            'orig_img': torch.tensor(np.asarray(img))
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def to_coco_label(self, label: int):
        return self.pred_to_coco[label]