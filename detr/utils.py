import random
from typing import List, Tuple
import torch 
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment

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


'''
boxes1: 矩形集合, [矩形数, 4 (xmin, ymin, xmax, ymax)]
boxes2: 矩形集合, [矩形数, 4 (xmin, ymin, xmax, ymax)]
'''
def calc_giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    ious, union = calc_iou(boxes1, boxes2)

    # 二つの矩形を包含する最小の矩形の面積を計算
    left_top = torch.minimum(
        boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    right_bottom = torch.maximum(
        boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    areas = width_height.prod(dim=2)

    return ious - (areas - union) / areas


'''
preds_class         : 検出矩形のクラス,
                      [バッチサイズ, クエリ数, 物体クラス数]
preds_box           : 検出矩形の位置と大きさ,
                      [バッチサイズ, クエリ数, 4 (x, y, w, h)]
targets             : ラベル
loss_weight_class   : コストを計算する際の分類コストの重み
loss_weight_box_l1  : コストを計算する際の矩形のL1コストの重み
loss_weight_box_giou: コストを計算する際の矩形のGIoUコストの重み
'''
@torch.no_grad()
def hungarian_match(preds_class: torch.Tensor,
                     preds_box: torch.Tensor, targets: dict,
                     loss_weight_class: float=1.0,
                     loss_weight_box_l1: float=5.0,
                     loss_weight_box_giou: float=2.0):
    bs, num_queries = preds_class.shape[:2]

    # コスト計算を全てのサンプル一括で計算するため、
    # 全てのサンプルの予測結果を一旦第0軸に並べる
    preds_class = preds_class.flatten(0, 1).softmax(dim=1)
    preds_box = preds_box.flatten(0, 1)

    # 予測結果と同様に全てのサンプルの正解ラベルを一旦第0軸に並べる
    targets_class = torch.cat([target['classes']
                               for target in targets])
    # 正解矩形の値を正規化された画像上の座標に変換
    targets_box = torch.cat(
        [target['boxes'] / target['size'].repeat(2)
         for target in targets])

    # 分類のコストは正解クラスの予測確率にマイナスをかけたもの
    # 正解クラスの予測確率が高ければ高いほどコストが小さくなる
    cost_class = -preds_class[:, targets_class]

    # 矩形回帰の1つ目のコストとなる予測結果と正解のL1誤差の計算
    cost_box_l1 = torch.cdist(
        preds_box, convert_to_xywh(targets_box), p=1)

    # 矩形回帰の2つ目のコストとなる予測結果と正解のGIoU損失の計算
    cost_box_giou = -calc_giou(
        convert_to_xyxy(preds_box), targets_box)

    cost = loss_weight_class * cost_class + \
        loss_weight_box_l1 * cost_box_l1 + \
        loss_weight_box_giou * cost_box_giou

    # 一括で計算していたコストをサンプル毎に分解するため軸を変更
    # 検出矩形の軸を分解して、
    # [バッチサイズ、クエリ数、全サンプルの正解数]という軸構成になる
    cost = cost.view(bs, num_queries, -1)

    # SciPyのlinear_sum_assignment関数を適用するためCPUへ転送
    cost = cost.to('cpu')

    # 各サンプルの正解矩形数を計算
    sizes = [len(target['classes']) for target in targets]

    indices = []
    # 第2軸を各サンプルの正解矩形数で分解し、バッチ軸でサンプルを
    # 指定することで、各サンプルのコスト行列を取得
    for batch_id, c in enumerate(cost.split(sizes, dim=2)):
        c_batch = c[batch_id]
        # ハンガリアンアルゴリズムにより予測結果と正解のマッチング
        # クエリのインデックスと正解のインデックスを得る
        pred_indices, target_indices = linear_sum_assignment(c_batch)
        indices.append((
            torch.tensor(pred_indices, dtype=torch.int64),
            torch.tensor(target_indices, dtype=torch.int64)))

    return indices


'''
indices: ハンガリアンアルゴリズムにより得られたインデックス
'''
def get_pred_permutation_index(
    indices: List[Tuple[torch.Tensor]]):
    # マッチした予測結果のバッチインデックスを1つの軸に並べる
    batch_indices = torch.cat(
        [torch.full_like(pred_indices, i)
         for i, (pred_indices, _) in enumerate(indices)])
    # マッチした予測結果のクエリインデックスを1つの軸に並べる
    pred_indices = torch.cat([pred_indices
                              for (pred_indices, _) in indices])

    return batch_indices, pred_indices


