"""Module with tests for `layers` module.

Run tests via `pytest src`
"""
import torch

from src.models import layers


def test_positional_encoding():
    """Tests `PositionalEncoding` class.
    """
    positional_encoding = layers.PositionalEncoding(d_model=64, max_len=32)
    assert not list(positional_encoding.parameters())
    dummy_inputs = torch.randn((16, 32, 64))
    dummy_outputs = positional_encoding(dummy_inputs)
    assert dummy_outputs.shape == dummy_inputs.shape
    assert torch.any(dummy_outputs != dummy_inputs)


def test_downsample_block():
    """Tests `Downsample` class.
    """
    downsample = layers.Downsample(16, 32)
    dummy_inputs = torch.randn((32, 16, 8, 8))
    dummy_outputs = downsample(dummy_inputs)
    assert dummy_outputs.shape == (32, 32, 4, 4)


def test_residual_block():
    """Tests `ResidualBlock` class.
    """
    residual = layers.ResidualBlock(16)
    dummy_inputs = torch.randn((32, 16, 8, 8))
    dummy_outputs = residual(dummy_inputs)
    assert dummy_outputs.shape == dummy_inputs.shape
