"""Definitions of neural network layers used by models.

Contains layers for which it is appropriate to share code across models, i.e.
positional encoding layer.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """Applies positional encoding by adding pre-computed values to the input.

    Based on code from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
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
            max_len (int, optional): Maximum length of the sequences.
            Defaults to 512.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        shift_values = torch.zeros(1, max_len, d_model, requires_grad=False)
        shift_values[0, :, 0::2] = torch.sin(position * div_term)
        shift_values[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('shift_values', shift_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.shift_values[:, :x.size(1)]
        return x


class ResidualBlock(nn.Module):
    """A simple convolution with resudual connection.
    """

    def __init__(self, channels: int):
        """A simple convolution with residual connection.

        Args:
            channels (int): Channels in input and output tensors.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forwards input tensor through the module. Input/output shapes match
        `torch.nn.Conv2d` input/output shapes.

        Args:
            x (Tensor): Input tensor.
        """
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
    """Simple 2x downsampling that uses strided convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Simple 2x downsampling that uses strided convolution. Input/output
        shapes match those of `torch.nn.Conv2d`.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels in the output image.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2)

    def forward(self, x):
        """Forwards inputs through the module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv(x)
        return x


class PositionEmbedding(nn.Module):
    """Residual block tower that produces position embeddings from encoded
    position.
    """

    def __init__(self, board_planes: int, num_blocks: int,
                 feature_channels: int, embedding_dim: int):
        """Residual block tower that produces position embeddings from an
        encoded position. Input tensor is shaped `(n, s, c, h, w)`, output
        is shaped `(n, s, d)`, where:
        - n - batch dimension
        - s - sequence length
        - c - number of planes per one position
        - h - height of the board
        - w - width of the board
        - d - dimension of the embeddings

        Args:
            board_planes (int): Number of planes per one board.
            num_blocks (int): Number of residual blocks in the tower.
            feature_channels (int): Channels in each of the residual
            blocks.
            embedding_dim (int): Dimension of the embeddings.
        """
        super().__init__()
        embedding_modules = []
        embedding_modules.append(
            nn.Conv2d(board_planes, feature_channels, 3, 1, 1))
        for _ in range(num_blocks):
            embedding_modules.append(ResidualBlock(feature_channels))
        embedding_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        embedding_modules.append(nn.Flatten())
        embedding_modules.append(nn.Linear(feature_channels, embedding_dim))
        self.body = nn.Sequential(*embedding_modules)

    def forward(self, x):
        """Forwards arguments through the module.

        Shape:
            input: (n, s, c, h, w)
            output: (n, s, d)

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        n, s, c, h, w = x.shape
        x = x.view(n * s, c, h, w)
        x = self.body(x)
        _ns, d = x.shape
        x = x.view(n, s, d)
        return x
