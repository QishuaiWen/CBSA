import math
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

from attn_layers import *


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.SyncBatchNorm(out_planes)
    )


class ConvPatchEmbed(nn.Module):
    """ Image to Patch Embedding using multiple convolutional layers
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        # return x, (Hp, Wp)
        return x


class ISTA(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.step_size = 0.1
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)
        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)
        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)
        return output


class Transformer(nn.Module):
    def __init__(self, attn, ffn, dim, depth, heads, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.depth = depth
        self.dim = dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, attn(dim, heads=heads, dim_head=dim_head)),
                        PreNorm(dim, ffn(dim, dim))
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            grad_x = attn(x) + x

            x = ff(grad_x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self, *, attn, ffn, image_size, patch_size, num_classes, dim, depth, heads, pool='cls', channels=3, dim_head=64,
            ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )
        self.to_patch_embedding = ConvPatchEmbed(img_size=image_size, embed_dim=dim, patch_size=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(attn, ffn, dim, depth, heads, dim_head)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.patch_size = patch_size
        self.dim = dim

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        # x = self.forward_features(img)
        # x = self.forward_attn(img, -1)
        feature_pre = x
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        feature_last = x
        return self.mlp_head(x)
    
    def interpolate_pos_encoding(self, w, h):
        feat_w = w // self.patch_size
        feat_h = h // self.patch_size
        num_patches = feat_w * feat_h
        N = self.pos_embedding.shape[1] - 1
        if num_patches == N and w == h:
            return self.pos_embedding
        class_pos_embed = self.pos_embedding[:, 0]
        patch_pos_embed = self.pos_embedding[:, 1:]
        dim = self.dim
        
        feat_w, feat_h = feat_w + 0.1, feat_h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(feat_w / math.sqrt(N), feat_h / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(feat_w) == patch_pos_embed.shape[-2] and int(feat_h) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward_features(self, img):
        with torch.no_grad():
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            
            pos_embedding = nn.Parameter(self.interpolate_pos_encoding(w, h))
            
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

            x += pos_embedding[:, :(n + 1)]

            x = self.transformer(x)
            return x
    
    def forward_attn(self, img, layer):
        with torch.no_grad():
            h, w = img.shape[2], img.shape[3]
            feat_h, feat_w = h // self.patch_size, w // self.patch_size
            img = img[:, :, :feat_h * self.patch_size, :feat_w * self.patch_size]
            
            pos_embedding = nn.Parameter(self.interpolate_pos_encoding(w, h))
            
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

            x += pos_embedding[:, :(n + 1)]

            for i, (attn, ff) in enumerate(self.transformer.layers):
                if i == layer:
                    return attn(x, return_attn=True)
                grad_x = attn(x) + x
                x = ff(grad_x)
            return x
    

def Ours_tiny(num_classes=1000):
    return VisionTransformer(
        attn=CBSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        dim_head=192 // 3
        )

def Ours_small(num_classes=1000):
    return VisionTransformer(
        attn=CBSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dim_head=384 // 6
        )

def Ours_base(num_classes=1000):
    return VisionTransformer(
        attn=Ours,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dim_head=768 // 12
        )

def Ours_large(num_classes=1000):
    return VisionTransformer(
        attn=Ours,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dim_head=1024 // 16
        )

def CRATE_tiny(num_classes=1000):
    return VisionTransformer(
        attn=MSSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        dim_head=192 // 3
        )

def CRATE_small(num_classes=1000):
    return VisionTransformer(
        attn=MSSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dim_head=384 // 6
        )

def ToST_tiny(num_classes=1000):
    return VisionTransformer(
        attn=TSSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=12,
        heads=3,
        dim_head=192 // 3
        )

def ToST_small(num_classes=1000):
    return VisionTransformer(
        attn=TSSA,
        ffn=ISTA, 
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dim_head=384 // 6
        )