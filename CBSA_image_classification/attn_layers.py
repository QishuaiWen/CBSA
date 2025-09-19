import torch
from torch import nn

from einops import rearrange, repeat


class TSSA(nn.Module):
    # https://github.com/RobinWu218/ToST/blob/main/tost_vision/tost.py
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        num_heads = heads
        
        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)

        self.qkv = nn.Linear(dim, dim, bias=False)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        
        self.to_out = nn.Linear(dim, dim)
        self.scale = dim_head ** -0.5
    
    def forward(self, x, return_attn=False):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        b, h, N, d = w.shape

        if return_attn:
            dots = w @ w.transpose(-1, -2)
            return self.attend(dots)
        
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temp'}
    
    
class MSSA(nn.Module):
    # https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate.py
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)


    def forward(self, x, return_attn=False):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        
        if return_attn:
            return attn

        out = torch.matmul(attn, w)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class CBSA(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, inner_dim, bias=False)
        
        self.step_x = nn.Parameter(torch.randn(heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(heads, 1, 1))
        
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        
        self.qkv = nn.Identity()
    
    def attention(self, query, key, value):        
        dots = (query @ key.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = attn @ value
        return out, attn

    def forward(self, x, return_attn=False):
        b, n, c = x.shape
        h = width = int(n ** 0.5)
        
        w = self.proj(x)
        self.qkv(w)
        rep = self.pool(w[:, :-1, :].reshape(b, h, width, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)

        w = w.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)
        rep = rep.reshape(b, 64, self.heads, self.dim_head).permute(0, 2, 1, 3)

        rep_delta, attn = self.attention(rep, w, w)
        
        if return_attn:
            return attn.transpose(-1, -2) @ attn
        
        rep = rep + self.step_rep * rep_delta
        
        x_delta, _ = self.attention(rep, rep, rep)  
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta
        
        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')
        return self.to_out(x_delta)