"""
Code modified from DETR tranformer:
https://github.com/facebookresearch/detr
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""

from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
        

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,
            drop_attention_rate: float = 0.,
        ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=drop_attention_rate,
        )
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
        

class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU,
            drop_attention_rate: float = 0.,
        ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, drop_attention_rate=drop_attention_rate)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x