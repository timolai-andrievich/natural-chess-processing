"""Module containing models that use transformer architecture as a backbone.
"""
from torch import nn
from . import layers


class PositionTransformer(nn.Module):
    """Model that uses position embeddings as sequence elements, and
    transformer for sequence processing.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            vocab_size: int,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            board_planes: int = 17,
            num_blocks: int = 12,
            feature_channels: int = 16):
        """Model that uses position embeddings as sequence elements, and
        transformer for sequence processing.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): The dimensionality of position embeddings.
            nhead (int): The number of heads in attention modules. Defaults to
            8.
            num_encoder_layers (int): The number of layers in the transformer
            encoder. There is no `num_decoder_layers` parameter because the
            model uses only encoder part of the transformer. Defaults to 6.
            dim_feedforward (int): The dimension of the feedforward network
            part of the transformer encoder. Defaults to 2048.
            dropout (float): The dropout probability. Defaults to 0.1.
            board_planes (int): The number of planes per board. Defaults to 17.
            num_blocks (int): The number of residual blocks used for position
            embedding. Defaults to 12.
            feature_channels: The width of residual blocks used for position
            embeddings. Defaults to 16.
        """
        super().__init__()
        self.embed = layers.PositionEmbedding(board_planes, num_blocks,
                                              feature_channels, d_model)

        self.pe = layers.PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """Forwards the input tensor through the module.

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
        x = self.pe(x)
        _n, s, _d = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(s,
                                                              device=x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.fc(x)
        return x
