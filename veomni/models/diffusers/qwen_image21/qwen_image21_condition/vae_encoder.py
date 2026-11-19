"""Single-frame encoder adapted from DiffSynth-Studio (Apache-2.0).
Source: https://github.com/modelscope/DiffSynth-Studio/blob/08fdc0da8f41426885d9432e9007e6e460a4c30d/diffsynth/models/qwen_image_21_vae.py
The decoder and tiling paths are omitted. Return raw moments for VeOmni caches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config


def get_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class QwenImage21AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class QwenImage21CausalConv3d(nn.Conv2d):
    """Single-frame spatial convolution with the released checkpoint layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int | int | int],
        stride: int | tuple[int | int | int] = 1,
        padding: int | tuple[int | int | int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )

        # Set up causal padding
        self._padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x):
        padding = list(self._padding)
        x = x.squeeze(2)  # Remove the temporal dimension
        x = F.pad(x, padding)
        x = super().forward(x)
        x = x.unsqueeze(2)  # Add the temporal dimension back
        return x


class QwenImage21RMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            t in str(x.dtype) for t in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=(1 if self.channel_first else -1)).to(
            x.dtype
        )
        return normalized * self.scale * self.gamma + self.bias


class QwenImage21Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2))
        if mode == "downsample3d":
            # Present in the released encoder checkpoint; single images have no temporal history.
            self.time_conv = QwenImage21CausalConv3d(dim, dim, (1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        return self.resample(x.squeeze(2)).unsqueeze(2)


class QwenImage21ResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu") -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = QwenImage21RMS_norm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        x = self.conv2(x)

        # Add residual connection
        return x + h


class QwenImage21AttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenImage21RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImage21MidBlock(nn.Module):
    """
    Middle block for QwenVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [QwenImage21ResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImage21AttentionBlock(dim))
            resnets.append(QwenImage21ResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x):
        # First residual block
        x = self.resnets[0](x)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x)

        return x


class QwenImage21ResidualDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = QwenImage21AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)

        return x + self.avg_shortcut(x_copy)


class QwenImage21Encoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in (1, *dim_mult)]
        scale = 1.0

        # init block
        self.conv_in = QwenImage21CausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    QwenImage21ResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(QwenImage21AttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(QwenImage21Resample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = QwenImage21MidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x):
        x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            x = layer(x)

        ## middle
        x = self.mid_block(x)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        x = self.conv_out(x)

        return x


def _patchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    # x shape: [batch_size, channels, frames, height, width]
    batch_size, channels, frames, height, width = x.shape

    # Ensure height and width are divisible by patch_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})")

    # Reshape to [batch_size, channels, frames, height//patch_size, patch_size, width//patch_size, patch_size]
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)

    # Rearrange to [batch_size, channels * patch_size * patch_size, frames, height//patch_size, width//patch_size]
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)

    return x


class QwenImage21VAEEncoder(ModelMixin, ConfigMixin):
    _keys_to_ignore_on_load_unexpected = [r"^decoder\.", r"^post_quant_conv\."]

    @register_to_config
    def __init__(
        self,
        base_dim=96,
        z_dim=64,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(False, True, True, True),
        dropout=0.0,
        is_residual=True,
        in_channels=4,
        patch_size=None,
    ):
        super().__init__()
        self.encoder = QwenImage21Encoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=2 * z_dim,
            dim_mult=list(dim_mult),
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = QwenImage21CausalConv3d(2 * z_dim, 2 * z_dim, 1)

    def forward(self, pixels):
        if pixels.ndim != 5 or pixels.shape[2] != 1:
            raise ValueError("Expected a single RGBA frame per sample in BCHTW layout.")
        if self.config.patch_size is not None:
            pixels = _patchify(pixels, self.config.patch_size)
        return self.quant_conv(self.encoder(pixels))
