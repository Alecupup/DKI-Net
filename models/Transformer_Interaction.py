# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False,
                 use_mlp_plus_knet=True,
                 enable_kernel_interaction=True,
                 num_kernels=8,
                 kernel_interact_dim=64,
                 kernel_interact_iters=1,
                 interact_gamma_min=0.0,
                 interact_gamma_max=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.use_mlp_plus_knet = use_mlp_plus_knet
        self.enable_kernel_interaction = enable_kernel_interaction
        self.num_kernels = num_kernels
        self.kernel_interact_iters = kernel_interact_iters

        self.interact_gamma_min = interact_gamma_min
        self.interact_gamma_max = interact_gamma_max

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # The KID-FFN block replaces the standard FFN; avoid registering
        # unused MLP parameters in that block.
        self.mlp = None if use_mlp_plus_knet else Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

        # ===== KNet替换核心代码：轻量级动态卷积核FFN（5x5动态核 + 深度可分离 + 瓶颈压缩） =====
        knet_bottleneck = max(dim // 4, 16)
        knet_hidden = max(dim // 2, 32)
        self.knet_dynamic_kernel_size = 5

        self.knet_reduce = nn.Conv2d(dim, knet_bottleneck, kernel_size=1, bias=False)
        self.knet_reduce_norm = nn.BatchNorm2d(knet_bottleneck)
        self.knet_dw = nn.Conv2d(knet_bottleneck, knet_bottleneck, kernel_size=3, stride=1, padding=1,
                                 groups=knet_bottleneck, bias=False)
        self.knet_dw_norm = nn.BatchNorm2d(knet_bottleneck)
        self.knet_expand = nn.Conv2d(knet_bottleneck, dim, kernel_size=1, bias=False)

        self.knet_kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, knet_hidden, kernel_size=1, bias=True),
            act_layer(),
            nn.Conv2d(knet_hidden, dim * self.knet_dynamic_kernel_size * self.knet_dynamic_kernel_size, kernel_size=1, bias=True)
        )

        # knet动态分支门控：去掉上下界，直接学习门控系数（便于观测）
        self.knet_gamma = nn.Parameter(torch.tensor(1e-2))
        self.knet_act = act_layer()
        self.knet_drop = nn.Dropout(drop)

        # ===== 轻量核交互（仅在指定block启用）：低维投影 + 1轮MHA + 门控 =====
        if self.enable_kernel_interaction:
            interact_heads = 4 if kernel_interact_dim % 4 == 0 else 1
            self.kernel_proj_in = nn.Linear(dim, kernel_interact_dim)
            self.kernel_proj_out = nn.Linear(kernel_interact_dim, dim)
            self.kernel_interact_attn = nn.MultiheadAttention(
                embed_dim=kernel_interact_dim,
                num_heads=interact_heads,
                dropout=drop,
                batch_first=True
            )
            self.kernel_interact_norm = nn.LayerNorm(kernel_interact_dim)
            self.interact_gamma_raw = nn.Parameter(torch.tensor(-4.595))  # sigmoid后约0.01

    def _knet_ffn(self, x, H, W):
        """
        轻量级KNet-FFN
        输入/输出: x [B, H*W, C]
        """
        B, L, C = x.shape
        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)

        # 静态轻量主分支：瓶颈 + 深度可分离卷积
        base = self.knet_reduce(feat)
        base = self.knet_reduce_norm(base)
        base = self.knet_act(base)
        base = self.knet_dw(base)
        base = self.knet_dw_norm(base)
        base = self.knet_act(base)
        base = self.knet_expand(base)

        # 动态核分支：每个样本每个通道生成一个kxk卷积核
        k = self.knet_dynamic_kernel_size
        dyn_kernel = self.knet_kernel_gen(feat).view(B * C, 1, k, k)
        if k > 1:
            dyn_kernel = dyn_kernel - dyn_kernel.mean(dim=(2, 3), keepdim=True)
            dyn_kernel = dyn_kernel / (dyn_kernel.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
        else:
            # 1x1核只有一个元素，不能做std归一化，否则会产生NaN。
            dyn_kernel = torch.tanh(dyn_kernel)

        base_reshape = base.view(1, B * C, H, W)
        dyn_out = F.conv2d(base_reshape, dyn_kernel, bias=None, stride=1, padding=k // 2, groups=B * C)
        dyn_out = dyn_out.view(B, C, H, W)

        # knet门控融合（无上下界，直接观测学习到的knet_gamma）
        out_2d = base + self.knet_gamma * dyn_out

        # 轻量核交互：仅最后一个block可开启，默认1轮迭代
        if self.enable_kernel_interaction:
            # 从特征中提取小数量语义核 [B, K, C]
            kernels = F.adaptive_avg_pool2d(feat, output_size=(self.num_kernels, 1)).squeeze(-1).transpose(1, 2)
            kernels = self.kernel_proj_in(kernels)  # [B, K, D]
            for _ in range(self.kernel_interact_iters):
                kernels = self.kernel_interact_norm(kernels)
                inter_out, _ = self.kernel_interact_attn(kernels, kernels, kernels)
                kernels = kernels + inter_out

            # 将交互后的核汇聚成通道调制向量并门控注入
            kernel_context = self.kernel_proj_out(kernels).mean(dim=1).view(B, C, 1, 1)
            interact_gamma = self.interact_gamma_min + (self.interact_gamma_max - self.interact_gamma_min) * torch.sigmoid(self.interact_gamma_raw)
            out_2d = out_2d + interact_gamma * kernel_context

        out = out_2d.view(B, C, L).transpose(1, 2).contiguous()
        out = self.knet_drop(out)
        return out

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)

        ffn_in = self.norm2(x)
        if self.use_mlp_plus_knet:
            # 最后一个block：SA + KNet
            ffn_out = self._knet_ffn(ffn_in, H, W)
        else:
            # 前面block：SA + MLP
            ffn_out = self.mlp(ffn_in)
        x = x + self.drop_path(ffn_out)

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x

class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=128, patch_size=16, in_chans=384, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False,
                 num_kernels=8,
                 kernel_interact_dim=64,
                 kernel_interact_iters=1,
                 interact_gamma_min=0.0,
                 interact_gamma_max=0.5):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 1, 1),
            Rearrange('b c h w -> b (h w) c', h = img_size, w = img_size),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        # 固定策略（无开关）：
        # block1: SA + MLP
        # block2: SA + MLP + KNet(含核交互)
        self.stage = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads, reso=img_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=0, norm_layer=norm_layer,
                use_mlp_plus_knet=(i == depth - 1),
                enable_kernel_interaction=(i == depth - 1),
                num_kernels=num_kernels,
                kernel_interact_dim=kernel_interact_dim,
                kernel_interact_iters=kernel_interact_iters,
                interact_gamma_min=interact_gamma_min,
                interact_gamma_max=interact_gamma_max)
            for i in range(depth)])
       
        self.norm = norm_layer(curr_dim)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.stage1_conv_embed(x)
        for blk in self.stage:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        _, _, c = x.shape
        x = x.permute(0, 2, 1).contiguous().view(B, c, h, w)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

### 224 models

def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1,2,21,1],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model

def mit(img_size=128, patch_size=4, in_chans=384, embed_dim=144, **kwargs):
    model = CSWinTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=embed_dim, depth=2, split_size=1, num_heads=4, mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model
