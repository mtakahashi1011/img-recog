from typing import List, Tuple
import torch
import torch.nn.functional as F

from detection.utils import get_pred_permutation_index, convert_to_xywh, convert_to_xyxy, calc_giou, hungarian_match


'''
preds            : 検出矩形のクラス,
                   [バッチサイズ, クエリ数, 物体クラス数]
targets          : ラベル
indices          : ハンガリアンアルゴリズムにより得られたインデックス
background_weight: 背景クラスの交差エントロピー誤差の重み
'''
def class_loss_func(preds: torch.Tensor, targets: dict,
                     indices: List[Tuple[torch.Tensor]],
                     background_weight: float):
    pred_indices = get_pred_permutation_index(indices)

    # 物体クラス軸の最後の次元が背景クラス
    background_id = preds.shape[2] - 1

    # 正解ラベルとなるテンソルの作成
    # [バッチサイズ、クエリ数]のテンソルを作成し、背景IDを設定
    targets_class = preds.new_full(
        preds.shape[:2], background_id, dtype=torch.int64)
    # マッチした予測結果の部分に正解ラベルとなる物体クラスIDを代入
    targets_class[pred_indices] = torch.cat(
        [target['classes'][target_indices]
         for target, (_, target_indices) in zip(targets, indices)])

    # 背景クラスの正解数が多く、正解数に不均衡が生じるため、
    # 背景クラスの重みを下げる
    weights = preds.new_ones(preds.shape[2])
    weights[background_id] = background_weight

    loss = F.cross_entropy(preds.transpose(1, 2),
                           targets_class, weights)

    return loss


'''
preds  : 検出矩形の位置と大きさ,
         [バッチサイズ, クエリ数, 4 (x, y, w, h)]
targets: ラベル
indices: ハンガリアンアルゴリズムにより得られたインデックス
'''
def box_loss_func(preds: torch.Tensor, targets: dict,
                   indices: List[Tuple[torch.Tensor]]):
    pred_indices = get_pred_permutation_index(indices)

    # マッチした予測結果を抽出
    preds = preds[pred_indices]

    # マッチした正解を抽出
    targets_box = torch.cat([
        target['boxes'][target_indices] / target['size'].repeat(2)
        for target, (_, target_indices) in zip(targets, indices)])

    # 0除算を防ぐために、最小値が1になるように計算
    num_boxes = max(1, targets_box.shape[0])

    # マッチした予測結果と正解でL1誤差を計算
    loss_l1 = F.l1_loss(preds, convert_to_xywh(targets_box),
                        reduction='sum') / num_boxes

    # マッチした予測結果と正解でGIoU損失を計算
    gious = calc_giou(convert_to_xyxy(preds), targets_box)
    loss_giou = (1 - gious.diag()).sum() / num_boxes

    return loss_l1, loss_giou


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
def _loss_func(preds_class: torch.Tensor, preds_box: torch.Tensor,
              targets: dict, loss_weight_class: float=1.0,
              loss_weight_box_l1: float=5.0,
              loss_weight_box_giou: float=2.0,
              background_weight: float=0.1):
    indices = hungarian_match(preds_class, preds_box, targets,
                               loss_weight_class, loss_weight_box_l1,
                               loss_weight_box_giou)

    loss_class = loss_weight_class * class_loss_func(
        preds_class, targets, indices, background_weight)
    loss_box_l1, loss_box_giou = box_loss_func(
        preds_box, targets, indices)
    loss_box_l1 = loss_weight_box_l1 * loss_box_l1
    loss_box_giou = loss_weight_box_giou * loss_box_giou

    return loss_class, loss_box_l1, loss_box_giou