import copy
import torch
from torch import nn 

class SelfAttention(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, qkv_bias: bool=False):
        super().__init__()

        assert dim_hidden % num_heads == 0
        self.num_heads = num_heads 
        dim_head = dim_hidden // num_heads 

        self.scale = dim_head ** -0.5
        
        self.proj_in = nn.Linear(dim_hidden, dim_hidden*3, bias=qkv_bias)

        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x: torch.Tensor):
        '''
        x: 入力特徴量 [バッチサイズ，特徴量，特徴量次元]
        自己アテンションなのでクエリ数とバリュー数は特徴量数と等しい
        out: 出力特徴量 [バッチサイズ，特徴量，特徴量次元]
        自己アテンションでは入力と出力の形状は同じ
        '''
        bs, ns = x.shape[:2]
        qkv: torch.Tensor = self.proj_in(x)
        qkv = qkv.view(bs, ns, 3, self.num_heads, -1) # [バッチサイズ，特徴量数，QKV，ヘッド数，ヘッド特徴量次元]
        qkv = qkv.permute(2, 0, 3, 1, 4) # [QKV，バッチサイズ，ヘッド数，特徴量数，ヘッド特徴量次元]
        q, k, v = qkv.unbind(dim=0) # [バッチサイズ，ヘッド数，特徴量数，ヘッド特徴量次元]
        attn: torch.Tensor = q.matmul(k.transpose(-2, -1))
        attn = (attn * self.scale).softmax(dim=-1)
        out = attn.matmul(v)
        out = out.permute(0, 2, 1, 3) # [バッチサイズ，特徴量数，ヘッド数，ヘッド特徴量次元]
        out = out.flatten(start_dim=2) # [バッチサイズ，特徴量数，特徴量次元]
        out = self.proj_out(out) # [バッチサイズ，特徴量数，特徴量次元]
        return out
    
class FNN(nn.Module):
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)
    
    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        x = self.fnn(x) + x 
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, img_size: int, patch_size: int, dim_hidden: int, 
                 num_heads: int, dim_feedforward: int, num_layers: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size 
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        dim_patch = 3*patch_size**2 # チャネル数xパッチサイズxパッチサイズ

        self.patch_embed = nn.Linear(dim_patch, dim_hidden)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim_hidden))

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)
    
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        # Patch-flattening 
        bs, c, h, w = x.shape
        assert h == self.img_size and w == self.img_size
        x = x.view(bs, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5) # [バッチ数，パッチ数，パッチ数，チャンネル数，パッチサイズ，パッチサイズ]
        
        # Fully-connected Layer, Class Embedding and Positional Embedding
        x = x.reshape(bs, (h//self.patch_size)*(w//self.patch_size), -1) # [バッチ数，総パッチ数，パッチ次元]
        x = self.patch_embed(x) # [バッチ数，総パッチ数，特徴量次元]
        # expand: Passing -1 as the size for a dimension means not changing the size of that dimension
        class_token = self.class_token.expand(bs, -1, -1) # [バッチ数，1，特徴量次元]
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed

        for layer in self.layers:
            x = layer(x)
        
        # Feature extraction based on class embedding
        x = x[:, 0] # [バッチ数，特徴量次元]
        x = self.norm(x) # Layer Normalization
        if return_embed:
            return x 
        # Fully-connected Layer
        x = self.linear(x) # [バッチ数，クラス数]
        return x
    
    def get_device(self):
        return self.linear.weight.device
    
    def copy(self):
        return copy.deepcopy(self)