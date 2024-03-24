import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from retinanet.utils import convert_to_xywh, calc_iou


'''
preds_class        : 検出矩形のクラス,
                     [バッチサイズ, アンカーボックス数, 物体クラス数]
preds_box          : 検出矩形のアンカーボックスからの誤差,
                     [バッチサイズ, アンカーボックス数,
                                4 (x_diff, y_diff, w_diff, h_diff)]
anchors            : アンカーボックス,
                     [アンカーボックス数, 4 (xmin, ymin, xmax, ymax)]
targets            : ラベル
iou_lower_threshold: 検出矩形と正解矩形をマッチさせるか決める下の閾値
iou_upper_threshold: 検出矩形と正解矩形をマッチさせるか決める上の閾値
'''
def loss_func(preds_class: torch.Tensor, preds_box: torch.Tensor, anchors: torch.Tensor, targets: dict,
              iou_lower_threshold: float=0.4, iou_upper_threshold: float=0.5):
    anchors_xywh = convert_to_xywh(anchors)

    # 画像毎に目的関数を計算
    loss_class = preds_class.new_tensor(0)
    loss_box = preds_class.new_tensor(0)
    for img_preds_class, img_preds_box, img_targets in zip(
            preds_class, preds_box, targets):
        # 現在の画像に対する正解矩形がないとき
        if img_targets['classes'].shape[0] == 0:
            # 全ての物体クラスの確率が0となるように
            # (背景として分類されるように)ラベルを作成
            targets_class = torch.zeros_like(img_preds_class)
            loss_class += sigmoid_focal_loss(
                img_preds_class, targets_class, reduction='sum')

            continue

        # 各画素のアンカーボックスと正解矩形のIoUを計算し、
        # 各アンカーボックスに対して最大のIoUを持つ正解矩形を抽出
        ious = calc_iou(anchors, img_targets['boxes'])[0]
        ious_max, ious_argmax = ious.max(dim=1)

        # 分類のラベルを-1で初期化
        # IoUが下の閾値と上の閾値の間にあるアンカーボックスは
        # ラベルを-1として損失を計算しないようにする
        targets_class = torch.full_like(img_preds_class, -1)

        # アンカーボックスとマッチした正解矩形のIoUが下の閾値より
        # 小さい場合、全ての物体クラスの確率が0となるようラベルを用意
        targets_class[ious_max < iou_lower_threshold] = 0

        # アンカーボックスとマッチした正解矩形のIoUが上の閾値より
        # 大きい場合、陽性のアンカーボックスとして分類回帰の対象にする
        positive_masks = ious_max > iou_upper_threshold
        num_positive_anchors = positive_masks.sum()

        # 陽性のアンカーボックスについて、マッチした正解矩形が示す
        # 物体クラスの確率を1、それ以外を0として出力するように
        # ラベルに値を代入
        targets_class[positive_masks] = 0
        assigned_classes = img_targets['classes'][ious_argmax]
        targets_class[positive_masks,
                      assigned_classes[positive_masks]] = 1

        # IoUが下の閾値と上の閾値の間にあるアンカーボックスについては
        # 分類の損失を計算しない
        loss_class += ((targets_class != -1) * sigmoid_focal_loss(
            img_preds_class, targets_class)).sum() / \
            num_positive_anchors.clamp(min=1)

        # 陽性のアンカーボックスが一つも存在しないとき
        # 矩形の誤差の学習はしない
        if num_positive_anchors == 0:
            continue

        # 各アンカーボックスにマッチした正解矩形を抽出
        assigned_boxes = img_targets['boxes'][ious_argmax]
        assigned_boxes_xywh = convert_to_xywh(assigned_boxes)

        # アンカーボックスとマッチした正解矩形との誤差を計算し、
        # ラベルを作成
        targets_box = torch.zeros_like(img_preds_box)
        # 中心位置の誤差はアンカーボックスの大きさでスケール
        targets_box[:, :2] = (
            assigned_boxes_xywh[:, :2] - anchors_xywh[:, :2]) / \
            anchors_xywh[:, 2:]
        # 大きさはアンカーボックスに対するスケールのlogを予測
        targets_box[:, 2:] = (assigned_boxes_xywh[:, 2:] / \
                              anchors_xywh[:, 2:]).log()

        # L1誤差とL2誤差を組み合わせたsmooth L1誤差を使用
        loss_box += F.smooth_l1_loss(img_preds_box[positive_masks],
                                     targets_box[positive_masks],
                                     beta=1 / 9)

    batch_size = preds_class.shape[0]
    loss_class = loss_class / batch_size
    loss_box = loss_box / batch_size

    return loss_class, loss_box