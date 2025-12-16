import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid
from einops import rearrange


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """grid_size: int of the grid height and width return: pos_embed:
    [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/
    or w/o cls_token)"""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int, 表示1D网格的长度
    return: pos_embed: [grid_size, embed_dim]
    """
    grid = np.arange(grid_size, dtype=np.float32)  # 生成1D网格
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # 基于网格生成正弦余弦位置嵌入
    return pos_embed
class PatchEmbed1D(nn.Module):
    """1D Sequence to Patch Embedding."""

    def __init__(
        self,
        num_points_latent,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        bias=True,
        with_pos_embed=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_points_patch = num_points_latent // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.with_pos_embed = with_pos_embed
        if self.with_pos_embed:
            pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_points_patch)
            self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        latent = self.proj(latent)  # [b*f, c, num_points_latent] => [b*f, embed_dim, num_points_patch]
        num_points_patch = latent.shape[-1]  # num_points_patch = num_points_latent // patch_size
        latent = rearrange(latent, 'b c n -> b n c')
        if self.norm is not None:
            latent = self.norm(latent)
        if self.with_pos_embed:
            if self.num_points_patch != num_points_patch:
                pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], num_points_patch)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed
            pos_embed = pos_embed.to(latent.dtype)
            latent = latent + pos_embed
        return latent  # [b*f, num_points_patch, embed_dim]

class GroupPatchEmbed1D(nn.Module):
    """1D Sequence to Patch Embedding by group. for motion DiT."""

    def __init__(
        self,
        num_points_latent=16,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        bias=True,
        with_pos_embed=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_points_patch = num_points_latent // patch_size
        
        self.conv1 = nn.Conv1d(in_channels, embed_dim, kernel_size=5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv1d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv3 = nn.Conv1d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)
        self.linear = nn.Linear(8, 16)
        # self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.with_pos_embed = with_pos_embed
        if self.with_pos_embed:
            pos_embed = get_1d_sincos_pos_embed(embed_dim, self.num_points_patch)
            self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        latent_1 = self.conv1(latent)  # 全身特征 [bf, embed_dim, 8]
        latent_2 = self.conv2(latent)  # 肢体特征 [bf, embed_dim, 16]
        latent_3 = self.conv3(latent)  # 细节特征 [bf, embed_dim, 16]
        latent = self.linear(latent_1) + latent_2 + latent_3  # [bf, embed_dim, 16]
        num_points_patch = latent.shape[-1]  # num_points_patch = num_points_latent // patch_size
        latent = rearrange(latent, 'b c n -> b n c')
        if self.norm is not None:
            latent = self.norm(latent)
        if self.with_pos_embed:
            if self.num_points_patch != num_points_patch:
                pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], num_points_patch)
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed
            pos_embed = pos_embed.to(latent.dtype)
            latent = latent + pos_embed
        return latent  # [b*f, num_points_patch, embed_dim]


class FrameEmbed1D(nn.Module):
    def __init__(self, num_frames, embed_dim):
        super().__init__()
        self.num_frames = num_frames
        pos = torch.arange(0, num_frames)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        num_frames = latent.shape[1]
        if self.num_frames != num_frames:
            pos = torch.arange(0, num_frames)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed.to(latent.dtype)
        return latent + pos_embed
