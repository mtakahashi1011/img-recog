import torch
from detr.utils import convert_to_xyxy


'''
preds_class: 検出矩形のクラス, [バッチサイズ, クエリ数, 物体クラス数]
preds_box  : 検出矩形の位置と大きさ, [バッチサイズ, クエリ数, 4 (x, y, w, h)]
    シグモイド関数により正規化された座標を表す
targets    : ラベル
include_bg : 分類結果に背景を含めるかどうかを表す真偽値
'''
@torch.no_grad()
def post_process(preds_class: torch.Tensor, preds_box: torch.Tensor, targets: list[dict], include_bg: bool=False):
    probs = preds_class.softmax(dim=2)

    if include_bg:
        scores, labels = probs.max(dim=2)
    else:
        scores, labels = probs[:, :, :-1].max(dim=2)

    boxes = convert_to_xyxy(preds_box)
    img_sizes = torch.stack([target['orig_size'] for target in targets])
    boxes[:, :, ::2] *= img_sizes[:, 0].view(-1, 1, 1)
    boxes[:, :, 1::2] *= img_sizes[:, 1].view(-1, 1, 1)
    return scores, labels, boxes