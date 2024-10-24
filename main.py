import math
import torch

from torch import nn

from einops import rearrange


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    

def Upsample(dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim, 3, padding=1),
    )


def DownSample(dim):
    return nn.Conv2d(dim, dim, 3, stride=2, padding=1)


class PosEmbedding(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        div_term = math.log(10000) / dim * 2
        vec = torch.arange(half_dim)
        self.register_buffer("inv_freq", torch.exp(-div_term * vec))
        self.linear1 = nn.Linear(dim, output_dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, t):
        emb = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, time_embed):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        x = x + time_embed[:, :,None, None]
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = ConvBlock(dim, dim_out, groups=groups)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_embed):
        time_embed = self.mlp(time_embed)
        h = self.block1(x, time_embed)
        h = self.block2(h, time_embed)
        return h + self.res_conv(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, dim, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.attn = nn.MultiheadAttention(dim, 1, bias=False, batch_first=True)
        
    def forward(self, x):
        _, _, height, width = x.shape
        h = x
        h = self.norm(h)
        h = rearrange(h, "b c h w -> b (h w) c")
        out = self.attn(h, h, h, need_weights=False)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h=height, w=width)
        return out + x


class ResnetBlockWithAttention(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, add_attn, groups=32):
        super().__init__()
        self.block = ResnetBlock(dim, dim_out, time_emb_dim, groups=groups)
        self.attn = AttentionBlock(dim_out, groups=groups) if add_attn else nn.Identity()

    def forward(self, x, time_embed):
        x = self.block(x, time_embed)
        x = self.attn(x)
        return x
    
class UNetEncoderBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, num_res_blocks, add_attn, is_last, groups=32):
        super().__init__()
        layers = [ResnetBlockWithAttention(dim, dim_out, time_emb_dim, add_attn, groups=groups)]
        for _ in range(num_res_blocks - 1):
            layers.append(ResnetBlockWithAttention(dim_out, dim_out, time_emb_dim, add_attn, groups=groups))
        self.blocks = nn.Sequential(*layers)
        self.downsample = DownSample(dim_out) if not is_last else nn.Identity()

    def forward(self, x, time_embed):
        activations = []
        for block in self.blocks:
            x = block(x, time_embed)
            activations.append(x)
        x = self.downsample(x)
        if not isinstance(self.downsample, nn.Identity):
            activations.append(x)
        return x, activations
    
class UNetDecoderBlock(nn.Module):
    def __init__(self, dim, dim_out, step_ahead_channels, time_emb_dim, num_res_blocks, add_attn, is_last, groups=32):
        super().__init__()
        layers = [ResnetBlockWithAttention(dim + dim_out, dim_out, time_emb_dim, add_attn, groups=groups)]
        for _ in range(num_res_blocks - 2):
            layers.append(ResnetBlockWithAttention(dim_out * 2, dim_out, time_emb_dim, add_attn, groups=groups))

        layers.append(ResnetBlockWithAttention(dim_out + step_ahead_channels, dim_out, time_emb_dim, add_attn, groups=groups))
        self.blocks = nn.Sequential(*layers)
        self.upsample = Upsample(dim_out) if not is_last else nn.Identity()

    def forward(self, x, hs, time_embed):
        for block in self.blocks:
            x = block(torch.cat((x, hs.pop()), dim=1), time_embed)
        x = self.upsample(x)
        return x
    

class BottleneckBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=32):
        super().__init__()
        self.block1 = ResnetBlock(dim, dim_out, time_emb_dim, groups=groups)
        self.attn = AttentionBlock(dim_out, groups=groups)
        self.block2 = ResnetBlock(dim_out, dim_out, time_emb_dim, groups=groups)

    def forward(self, x, time_embed):
        x = self.block1(x, time_embed)
        x = self.attn(x)
        x = self.block2(x, time_embed)
        return x


class UNet(nn.Module):
    def __init__(self, resolution, channels, channel_mults, num_res_blocks, attn_resolutions, num_groups=32):
        super().__init__()
        self.resolution = resolution
        self.pos_embedding = PosEmbedding(channels, output_dim=channels * 4)
        self.project_in = nn.Conv2d(3, channels, 3, padding=1)

        self.down_blocks = nn.ModuleList([])
        time_emb_dim = channels * 4
        in_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            is_last = i == len(channel_mults) - 1

            self.down_blocks.append(UNetEncoderBlock(in_channels, out_channels, time_emb_dim, num_res_blocks, add_attn=resolution in attn_resolutions, is_last=is_last, groups=num_groups))
            if not is_last:
                resolution //= 2
            in_channels = out_channels

        self.mid_block = BottleneckBlock(in_channels, in_channels, time_emb_dim, groups=num_groups)
        
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(len(channel_mults))):
            out_channels = channels * channel_mults[i]
            is_last = i == 0
            step_ahead_channels = channels * channel_mults[i - 1] if not is_last else channels
            self.up_blocks.append(UNetDecoderBlock(in_channels, out_channels, step_ahead_channels, time_emb_dim, num_res_blocks + 1, add_attn=resolution in attn_resolutions, is_last=is_last, groups=num_groups))
            if not is_last:
                resolution *= 2
            in_channels = out_channels

        self.final_block = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x, time):
        assert x.shape[2] == self.resolution
        assert x.shape[3] == self.resolution

        time_embed = self.pos_embedding(time)
        x = self.project_in(x)
        activations = [x]
        for block in self.down_blocks:
            x, hs = block(x, time_embed)
            activations.extend(hs)

        x = self.mid_block(x, time_embed)
        for block in self.up_blocks:
            x = block(x, activations, time_embed)
        assert len(activations) == 0
        return self.final_block(x)
