# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import use_fused_attn
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.jit import Final


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()  # Keep existing fused attention check

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,  # <-- Add mask here
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            weights = None  # We don't return attention weights when using fused attention
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            weights = attn.clone().detach()

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, float("-inf"))  # Mask out positions

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_weights:
            return x, weights
        return x


class RuledDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.x_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.r_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, r, r_to, y, cx, return_attn_weights=False):
        num_r_tokens = r.shape[1]
        cr = cx.clone()
        xshift_msa, xscale_msa, xgate_msa, xshift_mlp, xscale_mlp, xgate_mlp = self.x_adaLN_modulation(cx).chunk(
            6, dim=1
        )
        rshift_msa, rscale_msa, rgate_msa, rshift_mlp, rscale_mlp, rgate_mlp = self.r_adaLN_modulation(cr).chunk(
            6, dim=1
        )
        mod_x = modulate(self.norm1(x), xshift_msa, xscale_msa)
        mod_r = modulate(self.norm1(r), rshift_msa, rscale_msa)
        mod_r_to = modulate(self.norm1(r_to), rshift_msa, rscale_msa)
        mod_y = self.norm1(y)
        mod_yxrrto = torch.cat([mod_y, mod_x, mod_r, mod_r_to], dim=1)
        mask = self.build_attn_mask(mod_yxrrto.shape[1], num_r_tokens).to(x.device)
        if return_attn_weights:
            attn_out, attn_weights = self.attn(mod_yxrrto, mask, return_weights=True)
        else:
            attn_out = self.attn(mod_yxrrto, mask)
        attn_out_y, attn_out_x, attn_out_r, attn_out_r_to = (
            attn_out[:, :1],
            attn_out[:, 1 : -2 * num_r_tokens],
            attn_out[:, -2 * num_r_tokens : -num_r_tokens],
            attn_out[:, -num_r_tokens:],
        )
        x = x + xgate_msa.unsqueeze(1) * attn_out_x
        r = r + rgate_msa.unsqueeze(1) * attn_out_r
        r_to = r_to + rgate_msa.unsqueeze(1) * attn_out_r_to
        y = y + attn_out_y  # No gating for y
        mod_x = modulate(self.norm2(x), xshift_mlp, xscale_mlp)
        mod_r = modulate(self.norm2(r), rshift_mlp, rscale_mlp)
        mod_r_to = modulate(self.norm2(r_to), rshift_mlp, rscale_mlp)
        mod_y = self.norm2(y)
        mod_yxrrto = torch.cat([mod_y, mod_x, mod_r, mod_r_to], dim=1)
        mlp_out = self.mlp(mod_yxrrto)
        mlp_out_y, mlp_out_x, mlp_out_r, mlp_out_r_to = (
            mlp_out[:, :1],
            mlp_out[:, 1 : -2 * num_r_tokens],
            mlp_out[:, -2 * num_r_tokens : -num_r_tokens],
            mlp_out[:, -num_r_tokens:],
        )
        x = x + xgate_mlp.unsqueeze(1) * mlp_out_x
        r = r + rgate_mlp.unsqueeze(1) * mlp_out_r
        r_to = r_to + rgate_mlp.unsqueeze(1) * mlp_out_r_to
        y = y + mlp_out_y  # No gating for y
        if return_attn_weights:
            return x, r, r_to, y, attn_weights
        return x, r, r_to, y

    @staticmethod
    def build_attn_mask(total_tokens, causal_size):
        mask = torch.zeros(total_tokens, total_tokens)

        # 1) mod_y, mod_x, mod_r, mod_r_to attend to first 1+N+M tokens
        mask[:, :-causal_size] = 1

        # 3) mod_r_to attends to last M tokens causally
        causal_block = torch.tril(torch.ones(causal_size, causal_size))
        mask[-causal_size:, -causal_size:] = causal_block

        return mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RuleFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, activation=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class RuledDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        max_input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_msg_tokens=16,
        msg_tokens_dim=64,
        representation_layer=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_input_size = max_input_size
        self.patch_size = patch_size
        self.max_h, self.max_w = (
            max_input_size // patch_size,
            max_input_size // patch_size,
        )
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.embed_dim = hidden_size
        self.num_msg_tokens = num_msg_tokens
        self.msg_tokens_dim = msg_tokens_dim
        self.representation_layer = representation_layer

        self.x_embedder = PatchEmbed(max_input_size, patch_size, in_channels, hidden_size, bias=True)
        self.r_embedder = nn.Sequential(
            nn.Linear(msg_tokens_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.c_embedder = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        max_num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, hidden_size), requires_grad=False)
        self.y_pos_offset = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.r_pos_embed = nn.Parameter(torch.randn(1, num_msg_tokens, hidden_size), requires_grad=True)
        self.c_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.r_to_token = nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.r_to_pos_embed = nn.Parameter(torch.randn(1, num_msg_tokens, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([RuledDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.r_final_layer = RuleFinalLayer(hidden_size, msg_tokens_dim)
        self.c_final_layer = RuleFinalLayer(hidden_size, 2, activation=torch.tanh)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLPs:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.x_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.x_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.r_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.r_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def unflatten(self, x):
        """
        x: (N, T, C)
        out: (N, C, H, W)
        """

        c = x.shape[2]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, c).permute(0, 3, 1, 2)

        return x

    def set_fused_attn(self, use_fused: bool):
        """
        Enable or disable fused attention in all Attention layers.
        """
        for block in self.blocks:
            block.attn.fused_attn = use_fused

    def forward(self, x, r, tx, c, y, return_attn_weights=False, attn_weights_layer=-1):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        r: (N, L, D) tensor of msg tokens
        tx: (N,) tensor of diffusion timesteps
        c: (N, 2) tensor of offsets
        y: (N, C, H, W) tensor of spatial conditioning inputs (images or latent representations of images)
        """
        if return_attn_weights:
            self.set_fused_attn(False)
            assert attn_weights_layer is not None and 0 <= attn_weights_layer < len(
                self.blocks
            ), f"attn_weights_layer must be between 0 and {len(self.blocks)-1}"

        h, w = x.shape[2], x.shape[3]
        self.x_embedder.set_input_size((h, w))
        pos_embed = rearrange(self.pos_embed, "1 (h w) d -> 1 h w d", h=self.max_h, w=self.max_w)
        pos_embed = pos_embed[:, : h // self.patch_size, : w // self.patch_size, :]
        pos_embed = rearrange(pos_embed, "1 h w d -> 1 (h w) d")
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        r = self.r_embedder(r) + self.r_pos_embed
        r_to = self.r_to_token.expand(r.shape[0], -1, -1) + self.r_to_pos_embed
        c = self.c_embedder(c.unsqueeze(1)) + self.c_pos_embed
        if y is not None:
            assert (
                y.shape[2] == h and y.shape[3] == w
            ), f"y input size ({y.shape[2]}x{y.shape[3]}) doesn't match x input size ({h}x{w})"
            y = self.x_embedder(y) + pos_embed + self.y_pos_offset
            x = torch.cat([x, y], dim=1)  # (N, 2T, D)
        tx = self.t_embedder(tx)  # (N, D)
        for i, block in enumerate(self.blocks):
            if return_attn_weights and i == attn_weights_layer:
                x, r, r_to, c, attn_weights = block(x, r, r_to, c, tx, return_attn_weights=True)
            else:
                x, r, r_to, c = block(x, r, r_to, c, tx)  # (N, T[2T], D)
            if i == self.representation_layer:
                representation = self.unflatten(x[:, : self.x_embedder.num_patches])  # (N, T, D)
                representation_cls = r.clone()

        x = self.final_layer(x[:, : self.x_embedder.num_patches], tx)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        r_to = self.r_final_layer(r_to)
        c = self.c_final_layer(c)

        if return_attn_weights:
            self.set_fused_attn(True)
            return x, r_to, c, representation, representation_cls, attn_weights
        return x, r_to, c, representation, representation_cls


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
