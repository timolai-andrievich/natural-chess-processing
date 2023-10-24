"""Definitions of neural network layers used by models.

Contains layers for which it is logical to share code, i.e.
positional encoding layer.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """Applies positional encoding by adding pre-computed values to the input.

    Based on code from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Uses positional encoding formula from "Attention is All You Need":
    PE(pos, 2i) = sin(pos / 10^4^(2i/d_model))
    PE(pos, 2i + 1) = cos(pos / 10^4^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 512):
        """Initializes the positional encoding layer. To speed up
        computations, the encodings are precomputed, and the
        layer cannot be applied to sequences longer than `max_len`.
        Input shape: (`batch_size`, `sequence_len`, `d_model`)
        Output shape: the same as the input shape.

        Args:
            d_model (int): The dimension of the embedding vectors.
            max_len (int, optional): Maximum length of the sequences. Defaults to 512.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.shift_values = torch.zeros(1,
                                        max_len,
                                        d_model,
                                        requires_grad=False)
        self.shift_values[0, :, 0::2] = torch.sin(position * div_term)
        self.shift_values[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.shift_values[:, :x.size(1)]
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = shortcut + x
        x = F.relu(x)
        x = self.norm2(x)
        return x


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2)

    def forward(self, x):
        x = self.conv(x)
        return x
