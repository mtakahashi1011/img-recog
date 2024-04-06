import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_hidden, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # 自己アテンションブロックの構成要素
        self.self_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)

        # FNNブロックの構成要素
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos_encoding: Optional[Tensor]):
        return tensor if pos_encoding is None else tensor + pos_encoding

    def forward_post(self, x, mask: Optional[Tensor]=None, pos_encoding: Optional[Tensor]=None):
        q = k = self.with_pos_embed(x, pos_encoding)
        x2 = self.self_attn(q, k, value=x, key_padding_mask=mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

    def forward_pre(self, x, mask: Optional[Tensor]=None, pos_encoding: Optional[Tensor]=None):
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, pos_encoding)
        x2 = self.self_attn(q, k, value=x2, key_padding_mask=mask)[0]
        x = x + self.dropout1(x2)
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x

    def forward(self, x, mask: Optional[Tensor]=None, pos_encoding: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(x, mask, pos_encoding)
        return self.forward_post(x, mask, pos_encoding)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_hidden, num_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # 物体特徴量の自己アテンションブロックの構成要素
        self.self_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_hidden)
        
        # 物体特徴量と特徴マップの特徴量の交差アテンションブロックの構成要素
        self.multihead_attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_hidden)

        # FNNブロックの構成要素
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim_hidden)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, h, x,
                     mask: Optional[Tensor]=None,
                     pos_encoding: Optional[Tensor]=None,
                     query_embed: Optional[Tensor]=None):
        q = k = self.with_pos_embed(h, query_embed)
        tgt2 = self.self_attn(q, k, h)[0]
        h = h + self.dropout1(tgt2)
        h = self.norm1(h)

        h2 = self.multihead_attn(query=self.with_pos_embed(h, query_embed),
                                   key=self.with_pos_embed(x, pos_encoding),
                                   value=x, 
                                   key_padding_mask=mask)[0]
        h = h + self.dropout2(h2)
        h = self.norm2(h)

        h2 = self.linear2(self.dropout(self.activation(self.linear1(h))))
        h = h + self.dropout3(h2)
        h = self.norm3(h)
        return h

    def forward_pre(self, h, x,
                    mask: Optional[Tensor]=None,
                    pos_encoding: Optional[Tensor]=None,
                    query_embed: Optional[Tensor]=None):
        h2 = self.norm1(h)
        q = k = self.with_pos_embed(h2, query_embed)
        h2 = self.self_attn(q, k, h2)[0]
        h = h + self.dropout1(h2)

        h2 = self.norm2(h)
        h2 = self.multihead_attn(query=self.with_pos_embed(h2, query_embed),
                                   key=self.with_pos_embed(x, pos_encoding),
                                   value=x, 
                                   key_padding_mask=mask)[0]
        h = h + self.dropout2(h2)

        h2 = self.norm3(h)
        h2 = self.linear2(self.dropout(self.activation(self.linear1(h2))))
        h = h + self.dropout3(h2)
        return h

    def forward(self, h, x,
                mask: Optional[Tensor]=None,
                pos_encoding: Optional[Tensor]=None,
                query_embed: Optional[Tensor]=None):
        if self.normalize_before:
            return self.forward_pre(h, x, mask, pos_encoding, query_embed)
        return self.forward_post(h, x, mask, pos_encoding, query_embed)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_encoder_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])
        self.num_encoder_layers = num_encoder_layers
        self.norm = norm

    def forward(self, x, mask: Optional[Tensor]=None, pos_encoding: Optional[Tensor]=None):
        output = x
        for layer in self.layers:
            output = layer(output, mask=mask, pos_encoding=pos_encoding)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_decoder_layers, norm=None, return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        self.num_decoder_layers = num_decoder_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, h, x,
                mask: Optional[Tensor]=None,
                pos_encoding: Optional[Tensor]=None,
                query_embed: Optional[Tensor]=None):
        output = h
        intermediate = []
        for layer in self.layers:
            output = layer(output, x, mask=mask,
                           pos_encoding=pos_encoding, query_embed=query_embed)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            hs = torch.stack(intermediate)
            hs = hs.permute(0, 2, 1, 3)
        return output.unsqueeze(0)
    

class Transformer(nn.Module):
    def __init__(self, dim_hidden=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(dim_hidden) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(dim_hidden, num_heads, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(dim_hidden) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.dim_hidden = dim_hidden
        self.num_heads = num_heads

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, mask, query_embed, pos_encoding):
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        pos_encoding = pos_encoding.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        query_embed = query_embed.unsqueeze(1).expand(-1, bs, -1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        x = self.encoder(x, mask=mask, pos_encoding=pos_encoding)

        h = torch.zeros_like(query_embed)
        hs = self.decoder(h, x, mask=mask, pos_endcoding=pos_encoding, query_embed=query_embed)
        return hs