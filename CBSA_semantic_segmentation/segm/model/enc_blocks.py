"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout):
#         super().__init__()
#         assert dim % heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = heads
#         self.head_dim = dim // heads
#         self.scale = self.head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(int(self.head_dim ** 0.5), int(self.head_dim ** 0.5)))

#     def attention(self, query, key, value):
#         query = query * self.scale
#         attn = query @ key.transpose(-2, -1)
#         attn = attn.softmax(dim=-1)
#         out = attn @ value
#         return out, attn
    
#     def forward(
#             self,
#             X,
#             mask=None,
#     ):
        
#         B, N, C = X.shape
#         X_qkv = self.qkv(X).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         X_q, X_k, X_v = X_qkv.unbind(0)  # X_q is pooled

#         H = W = int(N ** 0.5)
#         Q = self.pool(X_q[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)  # 注意到这里是先pooling再分head的

#         X_k = X_k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         X_v = X_v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         Q = Q.reshape(B, int(self.head_dim ** 0.5) ** 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         Q_delta, attn = self.attention(Q, X_k, X_v)
#         Q = Q + Q_delta
#         X_delta, _ = self.attention(Q, Q, Q)

#         X = attn.transpose(-1, -2) @ X_delta

#         X = X.transpose(1, 2).reshape(B, N, C)
#         X = self.proj(X)


# #         B, N, C = X.shape

# #         X_qkv = self.qkv(X).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
# #         X_q, X_k, X_v = X_qkv.unbind(0)

# #         X, _ = self.attention(X_q, X_k, X_v)

# #         X = X.transpose(1, 2).reshape(B, N, C)
# #         X = self.proj(X)
        
#         return X


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y = self.attn(self.norm1(x), mask)
        # if return_attention:
        #     return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
