import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d


class BasicBlock(nn.Module):
    '''
    ResNet18における残差ブロック
    in_channels : 入力チャネル数
    out_channels: 出力チャネル数
    stride      : 畳み込み層のストライド
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int=1):
        super().__init__()

        ''''' 残差接続 '''''
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = FrozenBatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        ''''''''''''''''''''

        # strideが1より大きいときにスキップ接続と残差接続の高さと幅を
        # 合わせるため、別途畳み込み演算を用意
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels)
            )

    '''
    順伝播関数
    x: 入力, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, x: torch.Tensor):
        ''''' 残差接続 '''''
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        ''''''''''''''''''''

        if self.downsample is not None:
            x = self.downsample(x)

        # 残差写像と恒等写像の要素毎の和を計算
        out += x

        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    '''
    ResNet18モデル
    '''
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

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

    '''
    順伝播関数
    x: 入力, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
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
    
class PositionalEncoding:
    '''
    位置エンコーディング生成クラス
    eps        : 0で割るのを防ぐための小さい定数
    temperature: 温度定数
    '''
    def __init__(self, eps: float=1e-6, temperature: int=10000):
        self.eps = eps
        self.temperature = temperature

    '''
    位置エンコーディングを生成する関数
    x   : 特徴マップ, [バッチサイズ, チャネル数, 高さ, 幅]
    mask: 画像領域を表すマスク, [バッチサイズ, 高さ, 幅]
    '''
    @torch.no_grad()
    def generate(self, x: torch.Tensor, mask: torch.Tensor):
        # 位置エンコーディングのチャネル数は入力の半分として
        # x方向のエンコーディングとy方向のエンコーディングを用意し、
        # それらを連結することで入力のチャネル数に合わせる
        num_pos_channels = x.shape[1] // 2

        # 温度定数の指数を計算するため、2の倍数を用意
        dim_t = torch.arange(0, num_pos_channels, 2,
                             dtype=x.dtype, device=x.device)
        # sinとcosを計算するために値を複製
        # [0, 2, ...] -> [0, 0, 2, 2, ...]
        dim_t = dim_t.repeat_interleave(2)
        # sinとcosへの入力のの分母となるT^{2i / d}を計算
        dim_t /= num_pos_channels
        dim_t = self.temperature ** dim_t

        # マスクされていない領域の座標を計算
        inverted_mask = ~mask
        y_encoding = inverted_mask.cumsum(1, dtype=torch.float32)
        x_encoding = inverted_mask.cumsum(2, dtype=torch.float32)

        # 座標を0-1に正規化して2πをかける
        y_encoding = 2 * math.pi * y_encoding / \
            (y_encoding.max(dim=1, keepdim=True)[0] + self.eps)
        x_encoding = 2 * math.pi * x_encoding / \
            (x_encoding.max(dim=2, keepdim=True)[0] + self.eps)

        # 座標を保持するテンソルにチャネル軸を追加して、
        # チャネル軸方向にdim_tで割る
        # 偶数チャネルはsin、奇数チャネルはcosの位置エンコーディング
        y_encoding = y_encoding.unsqueeze(1) / \
            dim_t.view(num_pos_channels, 1, 1)
        y_encoding[:, ::2] = y_encoding[:, ::2].sin()
        y_encoding[:, 1::2] = y_encoding[:, 1::2].cos()
        x_encoding = x_encoding.unsqueeze(1) / \
            dim_t.view(num_pos_channels, 1, 1)
        x_encoding[:, ::2] = x_encoding[:, ::2].sin()
        x_encoding[:, 1::2] = x_encoding[:, 1::2].cos()

        encoding = torch.cat((y_encoding, x_encoding), dim=1)

        return encoding
    
class TransformerEncoderLayer(nn.Module):
    '''
    Transformerエンコーダ層
    dim_hidden     : 特徴量次元
    num_heads      : マルチヘッドアテンションのヘッド数
    dim_feedforward: FNNの中間特徴量次元
    dropout        : ドロップアウト率
    '''
    def __init__(self, dim_hidden: int=256, num_heads: int=8,
                 dim_feedforward: int=2048, dropout: float=0.1):
        super().__init__()

        # 自己アテンションブロックの構成要素
        self.self_attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)

        # FNNブロックの構成要素
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

    '''
    順伝播関数
    x           : 特徴マップの特徴量,
                  [特徴量数, バッチサイズ, 特徴量次元]
    pos_encoding: 位置エンコーディング,
                  [特徴量数, バッチサイズ, 特徴量次元]
    mask        : 画像領域かどうかを表すマスク,
                  [バッチサイズ, 特徴量数]
    '''
    def forward(self, x: torch.Tensor, pos_encoding: torch.Tensor,
                mask: torch.Tensor):
        # クエリとキーには位置エンコーディングを加算することで
        # アテンションの計算に位置の情報が使われるようにする
        q = k = x + pos_encoding

        # self_attenionにはクエリ、キー、バリューの順番に入力
        # key_padding_maskにmaskを渡すことでマスクが真の値を持つ領域の
        # キーは使われなくなり、特徴収集の対象から外れる
        # MutltiheadAttentionクラスは特徴収集結果とアテンションの値の
        # 2つの結果を返すが、特徴収集結果のみを使うので[0]とする
        x2 = self.self_attention(q, k, x, key_padding_mask=mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.fnn(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x

class TransformerDecoderLayer(nn.Module):
    '''
    Transformerデコーダ層
    dim_hidden     : 特徴量次元
    num_heads      : マルチヘッドアテンションのヘッド数
    dim_feedforward: FNNの中間特徴量次元
    dropout        : ドロップアウト率
    '''
    def __init__(self, dim_hidden: int=256, num_heads: int=8,
                 dim_feedforward: float=2048, dropout: float=0.1):
        super().__init__()

        # 物体特徴量の自己アテンションブロックの構成要素
        self.self_attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)

        # 物体特徴量と特徴マップの特徴量の
        # 交差アテンションブロックの構成要素
        self.cross_attention = nn.MultiheadAttention(
            dim_hidden, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

        # FNNブロックの構成要素
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim_hidden)

    '''
    順伝播関数
    h           : 物体特徴量, [クエリ数, バッチサイズ, 特徴量次元]
    query_embed : 物体クエリ埋め込み,
                  [クエリ数, バッチサイズ, 特徴量次元]
    x           : 特徴マップの特徴量,
                  [特徴量数, バッチサイズ, 特徴量次元]
    pos_encoding: 位置エンコーディング,
                  [特徴量数, バッチサイズ, 特徴量次元]
    mask        : 画像領域かどうかを表すマスク,
                  [バッチサイズ, 特徴量数]
    '''
    def forward(self, h: torch.Tensor, query_embed: torch.Tensor,
                x: torch.Tensor, pos_encoding: torch.Tensor,
                mask: torch.Tensor):
        # 物体クエリ埋め込みの自己アテンション
        q = k = h + query_embed
        h2 = self.self_attention(q, k, h)[0]
        h = h + self.dropout1(h2)
        h = self.norm1(h)

        # 物体クエリ埋め込みと特徴マップの交差アテンション
        h2 = self.cross_attention(h + query_embed, x + pos_encoding,
                                  x, key_padding_mask=mask)[0]
        h = h + self.dropout2(h2)
        h = self.norm2(h)

        h2 = self.fnn(h)
        h = h + self.dropout3(h2)
        h = self.norm3(h)

        return h
    
class Transformer(nn.Module):
    '''
    エンコーダ層とデコーダ層をまとめるTransformer
    dim_hidden        : 特徴量次元
    num_heads         : マルチヘッドアテンションのヘッド数
    num_encoder_layers: エンコーダ層の数
    num_decoder_layers: デコーダ層の数
    dim_feedforward   : FNNの特徴量次元
    dropout           : ドロップアウト率
    '''
    def __init__(self, dim_hidden: int=256, num_heads: int=8,
                 num_encoder_layers: int=3, num_decoder_layers: int=3,
                 dim_feedforward: int=2048, dropout: float=0.1):
        super().__init__()

        # 引数で指定された数だけエンコーダ層とデコーダ層を用意
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim_hidden, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                dim_hidden, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    '''
    順伝播関数
    x           : 特徴マップ, [バッチサイズ, チャネル数, 高さ, 幅]
    pos_encoding: 位置エンコーディング,
                  [バッチサイズ, チャネル数, 高さ, 幅]
    mask        : マスク, [バッチサイズ, 高さ, 幅]
    query_embed : 物体クエリ埋め込み, [クエリ数, 特徴量次元]
    '''
    def forward(self, x: torch.Tensor, pos_encoding: torch.Tensor,
                mask: torch.Tensor, query_embed: torch.Tensor):
        bs = x.shape[0]

        ''' 入力をTransformerに入力するための整形 '''
        
        # 特徴マップ:
        # [バッチサイス、チャネル数、高さ、幅]
        # -> [高さ * 幅、バッチサイズ、チャネル数]
        x = x.flatten(2).permute(2, 0, 1)
        
        # 位置エンコーディング:
        # [バッチサイス、チャネル数、高さ、幅]
        # -> [高さ*幅、バッチサイズ、チャネル数]
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)
        
        # マスク:
        # [バッチサイス、高さ、幅] -> [バッチサイズ、高さ*幅]
        mask = mask.flatten(1)
        
        # 物体クエリ埋め込み:
        #[クエリ数、チャネル数]
        # -> [クエリ数、バッチサイズ、チャネル数]
        query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)

        ''''''''''''''''''''''''''''''''''''''''''''

        # エンコーダ層を直列に適用
        for layer in self.encoder_layers:
            x = layer(x, pos_encoding, mask)

        # デコーダ層を直列に適用
        # 途中のデコーダ層の出力も保持
        hs = []
        h = torch.zeros_like(query_embed)
        for layer in self.decoder_layers:
            h = layer(h, query_embed, x, pos_encoding, mask)
            hs.append(h)

        # 第0軸を追加して各デコーダ層の出力を第0軸で連結
        hs = torch.stack(hs)
        # [デコーダ層数、バッチサイズ、クエリ数、チャネル数]にする
        hs = hs.permute(0, 2, 1, 3)

        return hs
    
class DETR(nn.Module):
    '''
    DETRモデル(ResNet18バックボーン)
    num_queries       : 物体クエリ埋め込みの数
    dim_hidden        : Transformerで処理する際の特徴量次元
    num_heads         : マルチヘッドアッテンションのヘッド数
    num_encoder_layers: Transformerエンコーダの層数
    num_decoder_layers: Transformerデコーダの層数
    dim_feedforward   : TransformerのFNNの中間特徴量次元
    dropout           : Transformer内でのドロップアウト率
    num_classes       : 物体クラス数
    '''
    def __init__(self, num_queries: int, dim_hidden: int,
                 num_heads: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, num_classes: int):
        super().__init__()

        self.backbone = ResNet18()

        # バックボーンネットワークの特徴マップのチャネル数を
        # 減らすための畳み込み層
        self.proj = nn.Conv2d(512, dim_hidden, kernel_size=1)

        self.transformer = Transformer(
            dim_hidden, num_heads, num_encoder_layers,
            num_decoder_layers, dim_feedforward, dropout)

        # 分類ヘッド
        # 背景クラスのために実際の物体クラス数に1を追加
        self.class_head = nn.Linear(dim_hidden, num_classes + 1)

        # 矩形ヘッド
        self.box_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 4),
        )

        self.positional_encoding = PositionalEncoding()

        # 物体クラス埋め込み
        self.query_embed = nn.Embedding(num_queries, dim_hidden)

    '''
    順伝播関数
    x   : 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    mask: 画像領域かどうかを表すマスク, [バッチサイズ, 高さ, 幅]
    '''
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # バックボーンネットワークから第5レイヤーの特徴マップを取得
        x = self.backbone(x)[-1]
        print("="*50)
        print("bacbone output")
        print(x.shape)

        # Transformer処理用に特徴マップのチャネル数を削減
        x = self.proj(x)
        print("="*50)
        print("projection output")
        print(x.shape)

        # 入力画像と同じ大きさを持つmaskを特徴マップの大きさにリサイズ
        # interpolate関数はbool型には対応していないため、一旦xと
        # 同じ型に変換
        mask = mask.to(x.dtype)
        mask = F.interpolate(
            mask.unsqueeze(1), size=x.shape[2:])[:, 0]
        mask = mask.to(torch.bool)
        print("="*50)
        print("resized mask")
        print(mask.shape)

        pos_encoding = self.positional_encoding.generate(x, mask)
        print("="*50)
        print("positional encoding")
        print(pos_encoding.shape)

        hs = self.transformer(
            x, pos_encoding, mask, self.query_embed.weight)
        print("="*50)
        print("transformer output")
        print(hs.shape)

        preds_class = self.class_head(hs)
        print("="*50)
        print("preds_class")
        print(preds_class.shape)
        preds_box = self.box_head(hs).sigmoid()
        print("="*50)
        print("preds_box")
        print(preds_box.shape)

        return preds_class, preds_box

    '''
    モデルパラメータが保持されているデバイスを返す関数
    '''
    def get_device(self):
        return self.backbone.conv1.weight.device