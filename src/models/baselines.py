"""Baseline model for the game generation task.
"""
import torch
from torch import nn
from . import layers


class Baseline(nn.Module):
    """Baseline model.

    Consists of embedding layer, one-layer LSTM, and fully connected layer.
    """

    def __init__(self, d_model: int, vocab_size: int):
        """Initializes the model.
        Input size: (`batch_size`, `seq_len`)
        Output size (`batch_size`, `seq_len`, `vocab_size`)

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model,
                            batch_first=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards inputs through the layer.
        """
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.linear(x)
        return x


class PositionBaseline(nn.Module):

    def __init__(self,
                 d_model: int,
                 vocab_size: int,
                 board_planes: int = 17,
                 num_blocks: int = 12,
                 feature_channels: int = 16):

        super().__init__()

        embedding_modules = []
        embedding_modules.append(
            nn.Conv2d(board_planes, feature_channels, 3, 1, 1))
        for _ in range(num_blocks):
            embedding_modules.append(layers.ResidualBlock(feature_channels))
        embedding_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        embedding_modules.append(nn.Flatten())
        embedding_modules.append(nn.Linear(feature_channels, d_model))

        self.embed = nn.Sequential(*embedding_modules)
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model,
                            batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        n, s, c, h, w = x.shape
        x = x.view(n * s, c, h, w)
        x = self.embed(x)
        _, d = x.shape
        x = x.view(n, s, d)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
