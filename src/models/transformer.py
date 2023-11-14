import torch
from torch import nn
from . import layers


class PositionTransformer(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
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
        n, s, c, h, w = x.shape
        x = x.view(n * s, c, h, w)
        x = self.embed(x)
        _, d = x.shape
        x = x.view(n, s, d)
        x = self.pe(x)
        mask = nn.Transformer.generate_square_subsequent_mask(s,
                                                              device=x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.fc(x)
        return x
