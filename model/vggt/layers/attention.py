# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
from turtle import forward
from typing_extensions import Self
import warnings
import torch

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = True
from xformers.ops import memory_efficient_attention, unbind


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        mask_attention=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.mask_attention = mask_attention

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            if self.mask_attention is not None:
                n_token = q.shape[2] # q: b h n d
                mask = torch.ones(n_token, n_token, device=q.device, dtype=bool)
                mask[:-self.mask_attention, -self.mask_attention:] = False
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            raise AssertionError('still using vanilla attention :-(')
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_with_kv_cache(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = 'save_cache' # save_cache or read_cache

    def forward(self, x: Tensor, pos=None) -> Tensor:
        if self.mode == 'save_cache':
            return self.forward_save_cache(x, pos)
        elif self.mode == 'read_cache':
            return self.forward_read_cache(x, pos)
        else:
            raise AssertionError('mode not supported')

    def forward_save_cache(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # save kv_cache
        # if hasattr(self, 'kv_cache'):
        #     del self.kv_cache
        
        if self.mask_attention is None:
            self.kv_cache = {
                'k': k,
                'v': v}
        else:
            self.kv_cache = {
                'k': k[:, :, :-self.mask_attention, :],
                'v': v[:, :, :-self.mask_attention, :]}
        print('cached kv', self.kv_cache['k'].shape, self.kv_cache['v'].shape)

        if self.fused_attn:
            if self.mask_attention is not None:
                n_token = q.shape[2] # q: b h n d
                mask = torch.ones(n_token, n_token, device=q.device, dtype=bool)
                mask[:-self.mask_attention, -self.mask_attention:] = False
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            raise AssertionError('still using vanilla attention :-(')
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def forward_read_cache(self, x: Tensor, pos=None) -> Tensor:
        assert hasattr(self, 'kv_cache'), 'kv_cache not saved yet'
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        assert q.shape[2] == self.mask_attention, 'only pass target view tokens when using kv_cache'

        # use kv_cache
        k = torch.cat([self.kv_cache['k'], k], 2)
        v = torch.cat([self.kv_cache['v'], v], 2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MemEffAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_vanilla = False
    
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        # assert pos is None

        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        ### mod
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        # x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
