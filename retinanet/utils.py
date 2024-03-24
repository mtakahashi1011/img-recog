import random
import torch 
from torch.utils.data import Dataset

def generate_subset(dataset: Dataset, ratio: float, random_seed: int=0):
    size = int(len(dataset)*ratio)
    indices = list(range(len(dataset)))
    random.seed(random_seed)
    random.shuffle(indices)
    indices1, indices2 = indices[:size], indices[size:]
    return indices1, indices2

def convert_to_xywh(boxes: torch.Tensor):
    wh = boxes[..., 2:] - boxes[..., :2]
    xy = boxes[..., :2] + wh / 2
    boxes = torch.cat((xy, wh), dim=-1)

    return boxes

def convert_to_xyxy(boxes: torch.Tensor):
    xymin = boxes[..., :2] - boxes[..., 2:] / 2
    xymax = boxes[..., 2:] + xymin
    boxes = torch.cat((xymin, xymax), dim=-1)

    return boxes

def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    # 第1軸をunsqueezeし、ブロードキャストを利用することで
    # [矩形数, 1, 2] と[矩形数, 2]の演算結果が
    # [boxes1の矩形数, boxes2の矩形数, 2] となる

    # 積集合の左上の座標を取得
    intersect_left_top = torch.maximum(
        boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    # 積集合の右下の座標を取得
    intersect_right_bottom = torch.minimum(
        boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])

    # 積集合の幅と高さを算出し、面積を計算
    intersect_width_height = (
        intersect_right_bottom - intersect_left_top).clamp(min=0)
    intersect_areas = intersect_width_height.prod(dim=2)

    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 和集合の面積を計算
    union_areas = areas1.unsqueeze(1) + areas2 - intersect_areas

    ious = intersect_areas / union_areas

    return ious, union_areas