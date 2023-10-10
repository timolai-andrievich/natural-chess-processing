"""Baseline model for the game generation task.
"""
import torch
from torch import nn

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
