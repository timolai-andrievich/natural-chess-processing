"""Tests the `transformer` module.

Run tests via `pytest src`.
"""
import torch
import torchtext

from src.models import transformer


def test_position_transformer():
    """Tests the position transformer model.
    """
    d_model = 8
    dummy_vocab = torchtext.vocab.vocab({'1': 1, '2': 2})
    model = transformer.PositionTransformer(d_model=d_model,
                                            vocab_size=len(dummy_vocab))
    dummy_inputs = torch.randn(16, 32, 17, 8, 8)
    outputs = model(dummy_inputs)
    assert outputs.shape == (16, 32, len(dummy_vocab))
