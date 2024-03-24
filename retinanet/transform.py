import random 
from PIL import Image
from typing import Sequence, Callable, Tuple, Union
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as Func


class RandomHorizontalFlip:
    '''
    無作為に画像を水平反転するクラス
    prob: 水平反転する確率
    '''
    def __init__(self, prob: float=0.5):
        self.prob = prob

    '''
    無作為に画像を水平反転する関数
    img   : 水平反転する画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        if random.random() < self.prob:
            # 画像の水平反転
            img = Func.hflip(img)

            # 正解矩形をx軸方向に反転
            # xmin, xmaxは水平反転すると大小が逆転し、
            # width - xmax, width - xminとなる
            width = img.size[0]
            target['boxes'][:, [0, 2]] = width - \
                target['boxes'][:, [2, 0]]

        return img, target


class RandomSizeCrop:
    '''
    無作為に画像を切り抜くクラス
    scale: 切り抜き前に対する切り抜き後の画像面積の下限と上限
    ratio: 切り抜き後の画像のアスペクト比の下限と上限
    '''
    def __init__(self, scale: Sequence[float],
                 ratio: Sequence[float]):
        self.scale = scale
        self.ratio = ratio

    '''
    無作為に画像を切り抜く関数
    img   : 切り抜きをする画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        width, height = img.size

        # 切り抜く領域の左上の座標と幅および高さを取得
        # 切り抜く領域はscaleとratioの下限と上限に従う
        top, left, cropped_height, cropped_width = \
            T.RandomResizedCrop.get_params(
                img, self.scale, self.ratio)

        # 左上の座標と幅および高さで指定した領域を切り抜き
        img = Func.crop(img, top, left, cropped_height, cropped_width)

        # 原点がx = left, y = topに移動し、合わせて矩形の座標も移動
        target['boxes'][:, ::2] -= left
        target['boxes'][:, 1::2] -= top

        # 矩形の座標が切り抜き後に領域外に出る場合は座標をクリップ
        target['boxes'][:, ::2] = \
            target['boxes'][:, ::2].clamp(min=0)
        target['boxes'][:, 1::2] = \
            target['boxes'][:, 1::2].clamp(min=0)
        target['boxes'][:, ::2] = \
            target['boxes'][:, ::2].clamp(max=cropped_width)
        target['boxes'][:, 1::2] = \
            target['boxes'][:, 1::2].clamp(max=cropped_height)

        # 幅と高さが0より大きくなる(矩形の面積が0でない)矩形のみ保持
        keep = (target['boxes'][:, 2] > target['boxes'][:, 0]) & \
            (target['boxes'][:, 3] > target['boxes'][:, 1])
        target['classes'] = target['classes'][keep]
        target['boxes'] = target['boxes'][keep]

        # 切り抜き後の画像の大きさを保持
        target['size'] = torch.tensor(
            [cropped_width, cropped_height], dtype=torch.int64)

        return img, target


class RandomResize:
    '''
    無作為に画像をアスペクト比を保持してリサイズするクラス
    min_sizes: 短辺の長さの候補、この中から無作為に長さを抽出
    max_size :  長辺の長さの最大値
    '''
    def __init__(self, min_sizes: Sequence[int], max_size: int):
        self.min_sizes = min_sizes
        self.max_size = max_size

    '''
    リサイズ後の短辺と長辺を計算する関数
    min_side: 短辺の長さ
    max_side: 長辺の長さ
    target  : 目標となる短辺の長さ
    '''
    def _get_target_size(self, min_side: int, max_side:int,
                         target: int):
        # アスペクト比を保持して短辺をtargetに合わせる
        max_side = int(max_side * target / min_side)
        min_side = target

        # 長辺がmax_sizeを超えている場合、
        # アスペクト比を保持して長辺をmax_sizeに合わせる
        if max_side > self.max_size:
            min_side = int(min_side * self.max_size / max_side)
            max_side = self.max_size

        return min_side, max_side

    '''
    無作為に画像をリサイズする関数
    img   : リサイズする画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        # 短辺の長さを候補の中から無作為に抽出
        min_size = random.choice(self.min_sizes)

        width, height = img.size

        # リサイズ後の大きさを取得
        # 幅と高さのどちらが短辺であるかで場合分け
        if width < height:
            resized_width, resized_height = self._get_target_size(
                width, height, min_size)
        else:
            resized_height, resized_width = self._get_target_size(
                height, width, min_size)

        # 指定した大きさに画像をリサイズ
        img = Func.resize(img, (resized_height, resized_width))

        # 正解矩形をリサイズ前後のスケールに合わせて変更
        ratio = resized_width / width
        target['boxes'] *= ratio

        # リサイズ後の画像の大きさを保持
        target['size'] = torch.tensor(
            [resized_width, resized_height], dtype=torch.int64)

        return img, target


class ToTensor:
    '''
    PIL画像をテンソルに変換する関数
    img   : テンソルに変換する画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        img = Func.to_tensor(img)

        return img, target


class Normalize:
    '''
    画像を標準化するクラス
    mean: R, G, Bチャネルそれぞれの平均値
    std : R, G, Bチャネルそれぞれの標準偏差
    '''
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    '''
    画像を標準化する関数
    img   : 標準化する画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: torch.Tensor, target: dict):
        img = Func.normalize(img, mean=self.mean, std=self.std)

        return img, target


class Compose:
    '''
    データ整形・拡張をまとめて適用するためのクラス
    transforms: データ整形・拡張のクラスインスタンスのシーケンス
    '''
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    '''
    データ整形・拡張を連続して適用する関数
    img   : データ整形・拡張する画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        for transform in self.transforms:
            img, target = transform(img, target)

        return img, target


class RandomSelect:
    '''
    2種類のデータ拡張を受け取り、無作為にどちらかを適用するクラス
    transform1: データ拡張1
    transform2: データ拡張2
    prob      : データ拡張1が適用される確率
    '''
    def __init__(self, transform1: Callable, transform2: Callable,
                 prob: float=0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.prob = prob

    '''
    データ拡張を無作為に選択して適用する関数
    img   : データ整形・拡張する画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img: Image, target: dict):
        if random.random() < self.prob:
            return self.transform1(img, target)

        return self.transform2(img, target)
    

def train_transform():
    min_sizes = (480, 512, 544, 576, 608)
    transforms = [
        RandomHorizontalFlip(),
        RandomSelect(
            RandomResize(min_sizes, max_size=1024),
            Compose([
                RandomSizeCrop(scale=(0.8, 1.0), ratio=(0.75, 1.333)),
                RandomResize(min_sizes, max_size=1024)
            ])
        ),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return Compose(transforms)


def valid_transform():
    min_sizes = (480, 512, 544, 576, 608)
    transforms = [
        RandomResize((min_sizes[-1],), max_size=1024),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return Compose(transforms)


# collate関数はデータセットクラスの__getitem__関数を使ってサンプルを取得したあとに，
# それらをミニバッチにまとめる作業を行う関数
# 画像の大きさが一定ではなかったり，ラベルが単純なものではない場合に指定する
def collate_func(batch: Sequence[Tuple[Union[torch.Tensor, dict]]]):
    max_height = 0
    max_width = 0
    for img, _ in batch:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    # バックボーンネットワークで特徴マップの解像度を下げるときに
    # 切り捨てが起きないように入力の幅と高さを32の倍数にしておく
    # もし32の倍数でない場合、バックボーンネットワークの特徴マップと
    # 特徴ピラミッドネットワークのアップスケーリングでできた特徴マップ
    # の大きさに不整合が生じ、加算できなくなる
    height = (max_height + 31) // 32 * 32
    width = (max_width + 31) // 32 * 32

    # 画像を一つにテンソルにまとめ、ラベルはリストに集約
    imgs = batch[0][0].new_zeros((len(batch), 3, height, width))
    targets = []
    for i, (img, target) in enumerate(batch):
        height, width = img.shape[1:]
        imgs[i, :, :height, :width] = img

        targets.append(target)

    return imgs, targets