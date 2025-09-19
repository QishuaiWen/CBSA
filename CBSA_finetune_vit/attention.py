from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F

from einops import rearrange, repeat


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.query = nn.Parameter(torch.randn(num_heads, 64, self.head_dim))
        # print(f"head_dim: {self.head_dim}")
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

#     def forward(
#             self,
#             x: torch.Tensor,
#             attn_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)

#         # if self.fused_attn:
#         #     x = F.scaled_dot_product_attention(
#         #         q, k, v,
#         #         attn_mask=attn_mask,
#         #         dropout_p=self.attn_drop.p if self.training else 0.,
#         #     )
#         # else:
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         # attn = maybe_add_mask(attn, attn_mask)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = attn @ v

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.norm(x)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

    def attention(self, query, key, value):
        query, key = self.q_norm(query), self.k_norm(key)
        query = query * self.scale
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ value
        return out, attn
        
#     def forward(
#             self,
#             x: torch.Tensor,
#             attn_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         B, N, C = x.shape
        
#         x_qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         x_q, x_k, x_v = x_qkv.unbind(0)
        
#         x, _ = self.attention(x_q, x_k, x_v)
        
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.norm(x)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
    
    def forward(
            self,
            X: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            layer_idx=None,
    ) -> torch.Tensor:
        # print(layer_idx)
        if layer_idx <= 100:
            # print(layer_idx)
            B, N, C = X.shape
            X_qkv = self.qkv(X).reshape(B, N, 3, C).permute(2, 0, 1, 3)
            X_q, X_k, X_v = X_qkv.unbind(0)  # X_q is pooled
            # print(X_q.shape)

            # Wq is leveraged
            # weight = self.qkv.weight
            # Wq, _, _ = torch.chunk(weight, 3, 0)
            # Q = rearrange(Wq, '(h k) d -> h k d', h=self.num_heads)
            # Q = Q @ Q.transpose(-1, -2)
            # Q = self.query
            # Q = Q.expand(B, -1, -1, -1)
            H = W = int(N ** 0.5)
            Q = self.pool(X_q[:, :-1, :].reshape(B, H, W, C).permute(0, 3, 1, 2)).reshape(B, C, -1).permute(0, 2, 1)  # 注意到这里是先pooling再分head的
            
            X_q = X_q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            X_k = X_k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            X_v = X_v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            Q = Q.reshape(B, 64, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # print("X_qkv&Q", X_q.shape, X_k.shape, X_v.shape, Q.shape)
            
            Q_delta, _ = self.attention(Q, X_k, X_v)
            # print("Q_delta", Q_delta.shape)
            # Q = Q + Q_delta
            # X_delta, _ = self.attention(Q, Q, Q)
            
            attn = ((X_q * self.scale) @ Q.transpose(-1, -2)).softmax(dim=-1)
            # print("attn", attn.shape)
            X = attn @ Q_delta
            # print("X", X.shape)
            # raise Exception("try done")
            
            X = X.transpose(1, 2).reshape(B, N, C)
            X = self.norm(X)
            X = self.proj(X)
            X = self.proj_drop(X)
        
        else:
            # print(layer_idx)
            B, N, C = X.shape

            X_qkv = self.qkv(X).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            X_q, X_k, X_v = X_qkv.unbind(0)

            X, _ = self.attention(X_q, X_k, X_v)

            X = X.transpose(1, 2).reshape(B, N, C)
            X = self.norm(X)
            X = self.proj(X)
            X = self.proj_drop(X)
            
        return X
    
    
# https://github.com/LeapLabTHU/FLatten-Transformer/blob/master/models/flatten_swin.py
# FocusedLinearAttention -> Attention
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, scale_norm=False, proj_bias=True, attn_drop=0., proj_drop=0., norm_layer=None):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, attn_mask=None):
       
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
#         q, k, v = qkv.unbind(0)
        
#         kernel_function = nn.ReLU()
#         q = kernel_function(q) + 1e-6
#         k = kernel_function(k) + 1e-6
        
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

#         z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
#         x = q @ kv * z

#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x