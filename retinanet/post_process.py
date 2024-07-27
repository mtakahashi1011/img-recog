import torch
from torchvision.ops import batched_nms

from retinanet.utils import convert_to_xywh, convert_to_xyxy


'''
preds_class   : 検出矩形のクラス,
                [バッチサイズ, アンカーボックス数, 物体クラス数]
preds_box     : 検出矩形のアンカーボックスからの誤差,
                [バッチサイズ, アンカーボックス数,
                                4 (x_diff, y_diff, w_diff, h_diff)]
anchors       : アンカーボックス,
                [アンカーボックス数, 4 (xmin, ymin, xmax, ymax)]
targets       : ラベル
conf_threshold: 信頼度の閾値
nms_threshold : NMSのIoU閾値
'''
@torch.no_grad()
def post_process(
        preds_class: torch.Tensor, 
        preds_box: torch.Tensor,
        anchors: torch.Tensor, targets: dict,
        conf_threshold: float=0.05,
        nms_threshold: float=0.5
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

    # アンカーボックスと予測誤差の結合
    anchors_xywh = convert_to_xywh(anchors)
    preds_box[:, :, :2] = anchors_xywh[:, :2] + preds_box[:, :, :2] * anchors_xywh[:, 2:]
    preds_box[:, :, 2:] = preds_box[:, :, 2:].exp() * anchors_xywh[:, 2:]
    preds_box = convert_to_xyxy(preds_box)

    # 物体クラスの予測確率をシグモイド関数で計算
    preds_class = preds_class.sigmoid()

    # forループで画像毎に処理を実施
    scores = []
    labels = []
    boxes = []
    for img_preds_class, img_preds_box, img_targets in zip(
            preds_class, preds_box, targets):
        # 検出矩形が画像内に収まるように座標をクリップ
        img_preds_box[:, ::2] = img_preds_box[:, ::2].clamp(min=0, max=img_targets['size'][0])
        img_preds_box[:, 1::2] = img_preds_box[:, 1::2].clamp(min=0, max=img_targets['size'][1])

        # 検出矩形は入力画像の大きさに合わせたものになっているので、元々の画像に合わせて検出矩形をスケール
        img_preds_box *= img_targets['orig_size'][0] / img_targets['size'][0]

        # 物体クラスのスコアとクラスIDを取得
        img_preds_score, img_preds_label = img_preds_class.max(dim=1)

        # 信頼度が閾値より高い検出矩形のみを残す
        keep = img_preds_score > conf_threshold
        img_preds_score = img_preds_score[keep]
        img_preds_label = img_preds_label[keep]
        img_preds_box = img_preds_box[keep]

        # クラス毎にNMSを適用
        keep_indices = batched_nms(img_preds_box, img_preds_score, img_preds_label, nms_threshold)

        scores.append(img_preds_score[keep_indices])
        labels.append(img_preds_label[keep_indices])
        boxes.append(img_preds_box[keep_indices])

    return scores, labels, boxes