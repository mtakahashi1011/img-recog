import copy
import torch
from torch import nn 
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim_hidden, num_heads)
        self.fnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)
    
    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        x = self.attention(x)[0] + x
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
        dim_patch = 3*patch_size**2 

        self.patch_embed = nn.Linear(dim_patch, dim_hidden)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim_hidden))

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)
    
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        bs, c, h, w = x.shape
        assert h == self.img_size and w == self.img_size
        x = x.view(bs, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5) # [B，H/P，W/P，C，P，P] 

        x = x.reshape(bs, (h//self.patch_size)*(w//self.patch_size), -1) # [B，Np，C*P^2] Np=H*W/P^2
        x = self.patch_embed(x) # [B，Np，D]

        class_token = self.class_token.expand(bs, -1, -1) # [B，1，D]
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed

        # EncoderLayer
        for layer in self.layers:
            x = layer(x)
        
        # Feature extraction based on class embedding
        x = x[:, 0] # [B，D]
        x = self.norm(x) 
        if return_embed:
            return x 
        x = self.linear(x) # [B，M]
        return x
    
    def get_device(self):
        return self.linear.weight.device
    
    def copy(self):
        return copy.deepcopy(self)