from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None 
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                FrozenBatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            x = self.downsample(x)

        out += x 
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5
    
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, num_features: int=256):
        super().__init__()
        self.levels = (3, 4, 5, 6, 7)
        self.p6 = nn.Conv2d(512, num_features, kernel_size=3, stride=2, padding=1)
        self.p7_relu = nn.ReLU(inplace=True)
        self.p7 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1)
        self.p5_1 = nn.Conv2d(512, num_features, kernel_size=1)
        self.p5_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.p4_1 = nn.Conv2d(256, num_features, kernel_size=1)
        self.p4_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(128, num_features, kernel_size=1)
        self.p3_2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        p6 = self.p6(c5)
        p7 = self.p7_relu(p6)
        p7= self.p7(p7)

        p5 = self.p5_1(c5)
        p5_up = F.interpolate(p5, scale_factor=2)
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4) + p5_up 
        p4_up = F.interpolate(p4, scale_factor=2)
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3) + p4_up
        p3 = self.p3_2(p3)
        return p3, p4, p5, p6, p7
    
class DetectionHead(nn.Module):
    def __init__(self, num_channels_per_anchor: int, num_anchors: int=9, num_features: int=256):
        super().__init__()
        self.num_anchors = num_anchors

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True))
            for _ in range(4)
        ])

        self.out_conv = nn.Conv2d(num_features, num_anchors*num_channels_per_anchor, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        for i in range(4):
            x = self.conv_blocks[i](x)
        x = self.out_conv(x)

        bs, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bs, w*h*self.num_anchors, -1)
        return x
    
class AnchorGenerator:
    def __init__(self, levels: int):
        ratios = torch.tensor([0.5, 1.0, 2.0])
        scales = torch.tensor([2**0, 2**(1/3), 2**(2/3)])
        self.num_anchors = ratios.shape[0] * scales.shape[0]
        self.strides = [2**level for level in levels]

        self.anchors = []
        for level in levels:
            base_length = 2**(level+2)
            scaled_lengths = base_length * scales 
            anchor_areas = scaled_lengths**2

            anchor_widths = (anchor_areas / ratios.unsqueeze(1)) ** 0.5
            anchor_heights = anchor_widths * ratios.unsqueeze(1)

            anchor_widths = anchor_widths.flatten()
            anchor_heights = anchor_heights.flatten()

            anchor_xmin = -0.5 * anchor_widths
            anchor_ymin = -0.5 * anchor_heights
            anchor_xmax = 0.5 * anchor_widths
            anchor_ymax = 0.5 * anchor_heights

            level_anchors = torch.stack(
                (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim=1
            )
            self.anchors.append(level_anchors)

    @torch.no_grad()
    def generate(self, feature_sizes: Sequence[torch.Size]):
        anchors = []
        for stride, level_anchors, feature_size in zip(
            self.strides, self.anchors, feature_sizes):
            height, width = feature_size
            xs = (torch.arange(width)+0.5) * stride 
            ys = (torch.arange(height)+0.5) * stride

            grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            anchor_xmin = (grid_x.unsqueeze(1) + level_anchors[:, 0]).flatten()
            anchor_ymin = (grid_y.unsqueeze(1) + level_anchors[:, 1]).flatten()
            anchor_xmax = (grid_x.unsqueeze(1) + level_anchors[:, 2]).flatten()
            anchor_ymax = (grid_y.unsqueeze(1) + level_anchors[:, 3]).flatten()

            level_anchors = torch.stack(
                (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim=1
            )

            anchors.append(level_anchors)
        anchors = torch.cat(anchors)
        return anchors 
    
class RetinaNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18()
        self.fpn = FeaturePyramidNetwork()
        self.anchor_generator = AnchorGenerator(self.fpn.levels)
        self.class_head = DetectionHead(
            num_channels_per_anchor=num_classes,
            num_anchors=self.anchor_generator.num_anchors
        )
        self.box_head = DetectionHead(
            num_channels_per_anchor=4,
            num_anchors=self.anchor_generator.num_anchors
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')

        # 分類ヘッドの出力にシグモイドを適用して各クラスの確率を出力
        # 学習開始時の確率が0.01になるようにパラメータを初期化
        prior = torch.tensor(0.01)
        nn.init.zeros_(self.class_head.out_conv.weight)
        nn.init.constant_(self.class_head.out_conv.bias,
                          -((1.0 - prior) / prior).log())

        # 学習開始時のアンカーボックスの中心位置の移動が0、
        # 大きさが1倍となるように矩形ヘッドを初期化
        nn.init.zeros_(self.box_head.out_conv.weight)
        nn.init.zeros_(self.box_head.out_conv.bias)

    def forward(self, x: torch.Tensor):
        cs = self.backbone(x)
        print("="*50)
        print("cs shape")
        for c in cs:
            print(c.shape)
        ps = self.fpn(*cs)
        print("="*50)
        print("ps shape")
        for p in ps:
            print(p.shape)

        preds_class = torch.cat(list(map(self.class_head, ps)), dim=1)
        print("="*50)
        print("preds_class shape")
        print(preds_class.shape)
        preds_box = torch.cat(list(map(self.box_head, ps)), dim=1)
        print("="*50)
        print("preds_box shape")
        print(preds_box.shape)

        feature_sizes = [p.shape[2:] for p in ps]
        print("="*50)
        print("feature_sizes")
        print(feature_sizes)
        anchors = self.anchor_generator.generate(feature_sizes)
        anchors = anchors.to(x.device)
        print("="*50)
        print("anchors shape")
        print(anchors.shape)
        return preds_class, preds_box, anchors 
    
    def get_device(self):
        return self.backbone.conv1.weight.device