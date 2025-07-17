import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from Deformable2d import DeformConv2d
import torch.nn.functional as F
from torchvision import models
from functools import partial
from enum import Enum
from torchvision.models import vit_b_16, ViT_B_16_Weights
from CBAM import cbam

def vit():
    # weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    # 3. 修改分类头为二分类
    num_classes = 2  # 你的类别数
    model.heads.head = nn.Sequential(
        nn.Linear(in_features=768, out_features=256),  # 原始特征维度768
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(256, num_classes)  # 最终输出2个类别
    )
    return model

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ResPatchEmbed(nn.Module):
    def __init__(self, img_size=56,patch_size=7,in_c = 256,embed_dim=147,norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=7)
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self,x):
        """
        x is feature layer of resnet
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.max_pool(x)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelectAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(SelectAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        scores = attn[:,:,0,:].mean(dim=1)   # to get the importance scores of cls between patches
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, scores
    

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x_attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SelectBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 rate = 0.5,
                 grid_h = 32,
                 grid_w = 32):
        super(SelectBlock, self).__init__()
        self.rate = rate
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.norm1 = norm_layer(dim)
        self.attn = SelectAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        self._initialize_weights()

    def _initialize_weights(self):
        """递归初始化所有子模块的权重"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """定义单个模块的初始化规则"""
        if isinstance(module, nn.Linear):
            # MLP 线性层初始化（Xavier Uniform）
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm 初始化（权重=1，偏置=0）
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, Attention):
            # 初始化 Attention 的 Q/K/V 和投影层
            if hasattr(module, 'qkv'):
                nn.init.xavier_uniform_(module.qkv.weight)
                if module.qkv.bias is not None:
                    nn.init.zeros_(module.qkv.bias)
            if hasattr(module, 'proj'):
                nn.init.xavier_uniform_(module.proj.weight)
                if module.proj.bias is not None:
                    nn.init.zeros_(module.proj.bias)

    def select_patches(self, x, scores,keep_indices_old=None):
        """
        根据CLS Token的注意力分数选择关键Patch,并返回其原图位置坐标
        参数:
            x: 输入特征 [B, N+1, d] (CLS Token + N个Patch)
            rate: 保留比例 (0~1)
            grid_h: 原图在高度方向的Patch数 (如14 for 224x224分块16x16)
            grid_w: 原图在宽度方向的Patch数
        
        返回:
            x_selected: 筛选后的特征 [B, 1+keep, d]
            positions: 选中Patch的原图坐标 [B, keep, 4] (x1,y1,x2,y2)
            keep_indices: 保留的Patch索引 [B, keep]
        """
        # 分离CLS和Patch特征
        cls_token = x[:, :1, :]  # [B, 1, d]
        patches = x[:, 1:, :]    # [B, N, d]
        scores = scores[:,1:]       # 排除第一个cls对自己的分数
        B, N, d = patches.shape
        patch_h = 224 / self.grid_h
        patch_w = 224 / self.grid_w

        keep_num = int(N * self.rate)
        _, keep_indices = torch.topk(scores, keep_num, dim=1,sorted=True)  # [B, keep_num]  keep_indices is the idx of patches in original img
        
        # select the patches
        patches_selected = torch.gather(patches,1, keep_indices.unsqueeze(-1).expand(-1, -1, d) )  # [B, keep_num, d]
        x_selected = torch.cat([cls_token, patches_selected], dim=1)
        # 如果存在上一层的全局索引，将当前层的局部索引映射回全局索引
        if keep_indices_old is not None:
            keep_indices = torch.gather(keep_indices_old, 1, keep_indices)

        # 找出在原图中的位置。
        # 将一维索引转换为二维网格坐标 [B, K]
        rows = keep_indices // self.grid_w  # 行坐标
        cols = keep_indices % self.grid_w   # 列坐标 
        self.grid_w
        # print(f'rows:{self.grid_w}')
        # print(f'rows:{rows},cols:{cols}')
        
        # 批量计算所有坐标 [B, K, 4]
        positions = torch.stack([
            cols * patch_w,        # x1
            rows * patch_h,        # y1
            (cols + 1) * patch_w,  # x2
            (rows + 1) * patch_h   # y2
        ], dim=-1).to(x.device)    # 形状 [B, K, 4]

        return x_selected, positions, keep_indices

    def forward(self, x,keep_indices_old=None):
        identity =x
        x,scores = self.attn(self.norm1(x))
        x = identity + self.drop_path(x)

        x_selected, positions, keep_indices = self.select_patches(x, scores, keep_indices_old)

        identity_selected = x_selected
        mlp_output = self.mlp(self.norm2(x_selected))
        x_selected = identity_selected + self.drop_path(mlp_output)  # 正确应用 DropPath
        return x_selected, positions, keep_indices



class SelectFormerBlock(nn.Module):
    def __init__(self,
                 embed_dim=7*7*3, depth=2, num_heads=1, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        
        super(SelectFormerBlock, self).__init__()
        """这个地方后面得改"""
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU  
        self.vit_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        # 初始化 ViT Blocks
        for i, block in enumerate(self.vit_blocks):
            self._init_vit_block(block, i)  # 显式调用块级初始化
        self.norm = norm_layer(embed_dim)
        
        self.select_block = SelectBlock(dim=embed_dim,num_heads=num_heads,rate=0.25)
    

    def _init_vit_block(self, block, block_idx):
        """正确初始化单个 ViT Block 的权重"""
        for m in block.modules():
            if isinstance(m, nn.Linear):
                # 初始化 MLP 线性层（Xavier Uniform）
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # 初始化 LayerNorm（权重=1，偏置=0）
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif hasattr(m, 'qkv'):  # 检查是否存在 qkv 属性（Attention 模块）
                # 初始化 Q/K/V 合并层（Xavier Uniform）
                nn.init.xavier_uniform_(m.qkv.weight)
                if m.qkv.bias is not None:
                    nn.init.zeros_(m.qkv.bias)
            elif hasattr(m, 'proj'):  # 检查是否存在 proj 属性（输出投影层）
                nn.init.xavier_uniform_(m.proj.weight)
                if m.proj.bias is not None:
                    nn.init.zeros_(m.proj.bias)


    def forward(self, x, keep_indices_old=None,return_pos=False):
        """x: (B, num_patches + cls_token, dim)"""
        x = self.vit_blocks(x) # 先经过一些vit blocks
        if return_pos:
            x_selected,pos, keep_indices = self.select_block(x,keep_indices_old = keep_indices_old) # 选择显著特征patch
            x_selected = self.norm(x_selected)
            # print(keep_indices)
            return x_selected,pos,keep_indices
        else:
            x_selected,_, keep_indices = self.select_block(x,keep_indices_old = keep_indices_old) # 选择显著特征patch
            x_selected = self.norm(x_selected)
            # print(keep_indices)
            return x_selected, keep_indices
    


class FeatureFusion(nn.Module):
    def __init__(self, img_size=56,patch_size=7,inc=256,
                 dim=147,    # 输入token的dim
                 num_heads=1,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 qkv_bias=False,
                 qk_scale=None):

        super(FeatureFusion,self).__init__()
        self.res_patch_embed = ResPatchEmbed(img_size=img_size, patch_size=patch_size, in_c=inc, embed_dim= 147)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.grid_h = img_size//patch_size
        self.grid_w = img_size//patch_size
        self.patch_size = patch_size
        # self.dcnv = DeformConv2d(inc=inc,outc=256-3,kernel_size=1,padding=0,stride=1)
        # self.conv = nn.Conv2d(inc,253,kernel_size=3,padding=1,stride=1)
        self.dcnv = DeformConv2d(inc=inc,outc=256,kernel_size=1,padding=0,stride=1)
        self.conv = nn.Conv2d(inc,256,kernel_size=3,padding=1,stride=1) 
        self.depthwise_separable_conv = nn.Sequential(
                                   nn.Conv2d(515, 515, kernel_size=3, stride=1, padding=1, groups=515),  # Depthwise
                                   nn.Conv2d(515, 256, kernel_size=1),                                 # Pointwise
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True)
                                )
                     
        self.cbam = cbam(in_channel=256)
        
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
        
    def cross_attention(self,cls_token,res_x):
        cls_token = cls_token.unsqueeze(1)
        cross_x = torch.cat([cls_token,res_x],dim=1)
        B, N, C = cross_x.shape
        
        qkv = self.qkv(cross_x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        scores = attn[:,:,0,:].mean(dim=1).squeeze(1)   # to get the importance scores of cls between patches

        attn = self.attn_drop(attn)

        # attn[:,:,0,:]为cls_token单独的注意力分数  ---> 得到[batch_size, num_heads, 1, embed_dim_per_head]---->[batch_size, 1, embed_dim]
        cls_attn = attn[:,:,0,:].unsqueeze(2)   # (B,num_heads,1,per_head_dim)
        x = (cls_attn @ v).transpose(1, 2).reshape(B, 1, C)        
        x = self.proj(x)
        token_cls = self.proj_drop(x)

        #################   为了得到分数最大的一块区域 #########################
        scores = scores[:,1:]       # 排除第一个cls对自己的分数
        _, keep_indices = torch.topk(scores, 1, dim=1,sorted=True)

        rows = keep_indices // self.grid_w  # 行坐标
        cols = keep_indices % self.grid_w   # 列坐标
        
        # 批量计算所有坐标 [B, K, 4]
        positions = torch.stack([
            cols * 7+ 3.5,        # x1
            rows * 7+ 3.5,        # y1
        ], dim=-1).to(x.device)    
        ######################################################################

        return positions,token_cls


    def forward(self,trans_x,res_x):
        """ 
        trans_x: output of transformer branch   shape:(B,cls+num_patches,dim)
        res_x:   output of resnet branch        shape:(B,C,H,W)
        assert: (H/patch_size)*(W/patch_size) = num_patches         patch_size*patch_size*C = dim
        """
        res_to_pacth = self.res_patch_embed(res_x)     # (B,num_patches,embed_dim)
        patches = trans_x[:, 1:, :]     # 需要将其转换为（B,C,H,W）
        trans_to_res = self.unpatchify(patches)       # (B,embed_dim,num_patches)

        cls_token = trans_x[:,0]        # get the cls_token of trans_x
        key_pos,cls_token = self.cross_attention(cls_token,res_to_pacth)        # 得到关键特征
        key_pos = key_pos.squeeze(1)  # (B,2)
        # print(key_pos.shape)
        trans_x = torch.cat([cls_token,patches],dim=1)
        res_x_conv = self.conv(res_x)
        res_x_dcnv, loss = self.dcnv(res_x,key_pos)        # 可变形卷积对特征进行聚集
        # res_x = res_x_conv + res_x_dcnv
        res_x = torch.cat([res_x_conv,res_x_dcnv],dim=1)
        res_x = torch.cat([res_x,trans_to_res],dim=1)     # 在通道维度进行拼接
        
        fusion_feat = self.depthwise_separable_conv(res_x)
        fusion_feat = self.cbam(fusion_feat)   # 添加注意力
        return fusion_feat,trans_x,loss

    
    
class TrainStage(Enum):
    STAGE1 = 1  # 仅训练ResNet
    STAGE2 = 2  # 仅训练Transformer
    STAGE3 = 3  # 训练FPN和融合模块

class ResNetBranch(nn.Module):
    """ResNet特征提取分支"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # self.features = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool,
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3,
        #     resnet.layer4
        # )
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv1(x)
        c2 = self.layer1(x)   # 输出shape: (B, 256, 56,56)
        c3 = self.layer2(c2)  # (B, 512, 28,28)
        c4 = self.layer3(c3)  # (B, 1024, 14,14)
        c5 = self.layer4(c4)  # (B, 2048, 7,7)
        
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}


class TransformerBranch(nn.Module):
    """Transformer特征处理分支"""
    def __init__(self, embed_dim: int, num_patches: int, pre_trans_blocks: int = 6,drop_ratio=0.):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        self.vit.heads = nn.Identity()
        self.vit.encoder.layers = self.vit.encoder.layers[:pre_trans_blocks]
        
        self.patch_embed = self.vit.conv_proj
        self.cls_token = self.vit.class_token
        self.pos_embed = self.vit.encoder.pos_embedding
        
        # 二次Patch处理
        self.patch_reembed = PatchEmbed(img_size=224, patch_size=7, in_c=3, embed_dim=embed_dim)
        self.cls_token_fc = nn.Linear(768, embed_dim)
        self.pos_embed_re = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed_re, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embed
        x = self.vit.encoder(x)

        cls_token = x[:, 0].unsqueeze(1)                     # [B,1,768]
        patches_feature = x[:, 1:, :]       # [B,196,768]
        trans_x = self.unpatchify(patches_feature)      # 还原回去
        trans_x = self.patch_reembed(trans_x)           # (B,3,224,224)--->(B,32*32,147)
        cls_token = self.cls_token_fc(cls_token)
        trans_x = torch.cat((cls_token, trans_x), dim=1)     # (B,32*32,147)-->(B,32*32+1,147)
        trans_x = self.pos_drop(trans_x + self.pos_embed_re)  # add pos_embed
        
        return trans_x


class PFDCNet(nn.Module):
    def __init__(
        self, 
        num_classes: int = 2, 
        embed_dim: int = 147, 
        num_patches: int = 32 * 32,
        pre_trans_blocks: int = 7,
        drop_ratio: float = 0.
    ):
        super().__init__()
        # 阶段标记初始化
        self.train_stage = TrainStage.STAGE1
        # 初始化各分支
        self.resnet_branch = ResNetBranch(pretrained=True)
        self.transformer_branch = TransformerBranch(embed_dim, num_patches, pre_trans_blocks,drop_ratio=drop_ratio)
    
        # 构建SelectFormer模块
        self.select_blocks = self._build_select_former_blocks(embed_dim)
       
        self.stage_weight = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))  # 初始值差异
        self.loss_weight = nn.Parameter(torch.tensor(-1.0))  # 高斯分布初始化
        
        # 特征融合模块
        self.fusion_modules = self._build_fusion_modules()
        
        # FPN头
        self.fpn_head = self._build_fpn_head()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        # 分类头
        self.classifiers = nn.ModuleDict({
            'stage1': nn.Linear(2048, num_classes),
            'stage2': nn.Sequential(
                        nn.LayerNorm(embed_dim*2),
                        nn.Linear(in_features=embed_dim*2, out_features=256),  # 原始特征维度147
                        nn.ReLU(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(256, num_classes)  # 最终输出2个类别
                        ),
            
            'stage3': nn.Sequential(
                        nn.Linear(in_features=1024, out_features=2),  # 原始特征维度147
                        #nn.ReLU(),
                        # nn.Dropout(p=0.1),
                        #nn.Linear(512, num_classes)  # 最终输出2个类别
                        ),
        })
    
    def _build_select_former_blocks(self, embed_dim: int) -> nn.ModuleList:
        """创建SelectFormer块序列"""
        return nn.ModuleList([
            SelectFormerBlock(depth=2, embed_dim=embed_dim)
            for i in range(5)  # 5个下采样块
        ])
    
    def _build_fusion_modules(self) -> nn.ModuleDict:
        """工厂方法创建特征融合模块"""
        return nn.ModuleDict({
            'fusion1': FeatureFusion(img_size=56, inc=256),
            'fusion2': FeatureFusion(img_size=28, inc=512),
            'fusion3': FeatureFusion(img_size=14, inc=1024),
            'fusion4': FeatureFusion(img_size=7, inc=2048)
        })
    
    def _build_fpn_head(self) -> nn.ModuleDict:
        """构建FPN头部组件"""
        return nn.ModuleDict({
            # 'toplayer': nn.Conv2d(2048, 256, kernel_size=1),
            # 'latlayer1': nn.Conv2d(1024, 256, kernel_size=1),
            # 'latlayer2': nn.Conv2d(512, 256, kernel_size=1),
            # 'latlayer3': nn.Conv2d(256, 256, kernel_size=1),
            'smooth': nn.Conv2d(256, 256, kernel_size=3, padding=1)
        })
    
    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """特征图上采样并相加"""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear') + y
    
    def forward(self, x: torch.Tensor, stage: TrainStage = None) -> tuple:
        """前向传播入口"""
        stage = stage or self.train_stage
        if stage == TrainStage.STAGE1:
            return self._forward_stage1(x)
        elif stage == TrainStage.STAGE2:
            return self._forward_stage2(x)
        else:
            return self._forward_stage3(x)
    
    def _forward_stage1(self, x: torch.Tensor) -> tuple:
        """阶段1:仅ResNet分支"""
        features = self.resnet_branch(x)
        pooled = self.resnet_branch.avgpool(features)
        logits = self.classifiers['stage1'](pooled.flatten(1))
        return logits, 0,torch.sigmoid(self.alpha)
    
    def _forward_stage2(self, x: torch.Tensor) -> tuple:
        """阶段2:仅Transformer分支"""
        trans_feat = self.transformer_branch(x)
        t1,_ = self.select_blocks[0](trans_feat)   # (B,32*32,147) --->(B,16*16,147)
        t2,_ = self.select_blocks[1](t1)       # (B,16*16,147)  --->(B,8*8,147)
        t3,_ = self.select_blocks[2](t2)       # (B,8*8,147)  --->(B,4*4,147)
        t4,_ = self.select_blocks[3](t3)
        t5,_ = self.select_blocks[4](t4)

        
        t6 = torch.cat((t5[:, 0],t5[:,1]),dim=1)
        # [处理transformer特征...]
        logits = self.classifiers['stage2'](t6)

        # logits = self.classifiers['stage2'](trans_feat[:,0])
        return logits, 0,0
    
    def _forward_stage3(self, x: torch.Tensor) -> tuple:
        """阶段3:融合所有分支"""
        # 获取ResNet特征
        resnet_feats = self.resnet_branch(x)
        
        # 处理Transformer分支
        trans_feat = self.transformer_branch(x)
        
        t1,keep_indices_old = self.select_blocks[0](trans_feat)   # (B,32*32,147) --->(B,16*16,147)
        # print(keep_indices_old)
        t2,keep_indices_old = self.select_blocks[1](t1,keep_indices_old)       # (B,16*16,147)  --->(B,8*8,147)
        # print(keep_indices_old)
        fusfeat1,t2,loss1 = self.fusion_modules['fusion1'](t2,resnet_feats['c2'])     # fusfeat1:(256,56,56)
        t3,keep_indices_old = self.select_blocks[2](t2,keep_indices_old)       # (B,8*8,147)  --->(B,4*4,147)
        # print(keep_indices_old)
        fusfeat2,t3,loss2 = self.fusion_modules['fusion2'](t3,resnet_feats['c3'])     # fusfeat2:(512,28,28)
        t4,keep_indices_old = self.select_blocks[3](t3,keep_indices_old)
        # print(keep_indices_old)
        fusfeat3,t4,loss3 = self.fusion_modules['fusion3'](t4,resnet_feats['c4'])     # fusfeat3:(1024,14,14)
        t5,final_pos,final_idx = self.select_blocks[4](t4,keep_indices_old,return_pos=True)
        fusfeat4,t5,loss4 = self.fusion_modules['fusion4'](t5,resnet_feats['c5'])     # fusfeat4:(2048,7,7)

        w1 = torch.exp(self.stage_weight[0]) / torch.sum(torch.exp(self.stage_weight))
        w2 = torch.exp(self.stage_weight[1]) / torch.sum(torch.exp(self.stage_weight))
        w3 = torch.exp(self.stage_weight[2]) / torch.sum(torch.exp(self.stage_weight))
        w4 = torch.exp(self.stage_weight[3]) / torch.sum(torch.exp(self.stage_weight))
        
        loss_weight = torch.exp(self.loss_weight)

        total_loss = loss_weight*sum(w * loss for w, loss in zip([w1,w2,w3,w4], [loss1, loss2, loss3, loss4]))
        
        # FPN计算
        p3 = self._upsample_add(fusfeat4, fusfeat3)
        p2 = self._upsample_add(p3, fusfeat2)
        p1 = self._upsample_add(p2, fusfeat1)
        
        pool1 = self.resnet_branch.avgpool(p1).flatten(1)
        pool2 = self.resnet_branch.avgpool(p2).flatten(1)
        pool3 = self.resnet_branch.avgpool(p3).flatten(1)
        pool4 = self.resnet_branch.avgpool(fusfeat4).flatten(1)
        out = torch.cat((pool1,pool2,pool3,pool4),dim=1)
        # 分类输出
        logits = self.classifiers['stage3'](out)
        
        return logits, total_loss,loss_weight,final_idx,final_pos   # 后面两个参数是为了画图需要，实际训练的时候直接删掉
    
    
    def set_train_stage(self, stage: TrainStage):
        """设置当前训练阶段"""
        self.train_stage = stage
    
    def freeze_parameters(self):
        """冻结非当前阶段参数"""
        for param in self.parameters():
            param.requires_grad = False
        
        if self.train_stage == TrainStage.STAGE1:
            for module in[self.resnet_branch,self.classifiers]:
                for param in module.parameters():
                    param.requires_grad = True
        elif self.train_stage == TrainStage.STAGE2:
            for module in [self.transformer_branch,self.select_blocks,self.classifiers]:
                for param in module.parameters():
                    param.requires_grad = True
        else:
            self.stage_weight.requires_grad=True
            self.loss_weight.requires_grad =True
            for module in [self.resnet_branch,self.fusion_modules, self.fpn_head,self.classifiers]:
                for param in module.parameters():
                    param.requires_grad = True


def visualize_selected_patches(image, position_tensor):
    """
    image: 原始图像 [H,W,C]
    position_tensor: 选中Patch的坐标 [keep_num, 4]
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if isinstance(position_tensor, torch.Tensor):
        position_tensor = position_tensor.cpu().numpy()
    else:
        position_tensor = position_tensor # 如果它已经是 NumPy 数组，则直接使用
    
    for box in position_tensor:
        x1, y1, x2, y2 = box
        
        rect = patches.Rectangle(
            (x1, y1),  # 左上角坐标
            x2 - x1,   # 宽度
            y2 - y1,   # 高度
            linewidth=2, 
            edgecolor='r',  # 红色边框
            facecolor='none'  # 无填充
        )
        ax.add_patch(rect)
    plt.show()