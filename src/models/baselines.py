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
    """LSTM model that embeds positions instead of moves.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            d_model: int,
            vocab_size: int,
            board_planes: int = 17,
            num_blocks: int = 12,
            feature_channels: int = 16):
        """LSTM model that uses position embeddings instead of move embeddings

        Args:
            d_model (int): Number of features in embeddings.
            vocab_size (int): Vocabulary size.
            board_planes (int): Number of planes per one position in
            model inputs.
            num_blocks (int): The number of residual blocks in the embedding
            part of the model.
            feature_channels (int): The width of residual blocks in the
            embedding part of the model.
        """

        super().__init__()

        self.embed = layers.PositionEmbedding(board_planes, num_blocks,
                                              feature_channels, d_model)
        self.lstm = nn.LSTM(input_size=d_model,
                            hidden_size=d_model,
                            batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input through the model.

        Args:
            x (Tensor): Input tensor, shape `(n, s, c, h, w)`, where:
            - `n` - batch dimension
            - `s` - sequence length
            - `c` - channels dimension (should be equal to `board_planes`)
            - `h` - the height of the board (should be 8)
            - `w` - the width of the board (should be 8)

        Returns:
            Tensor: Output tensor.
        """
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
