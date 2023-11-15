"""Tests the `baselines` module.

Run tests via `pytest src`.
"""
import torch
import torchtext

from src.models import baselines


def test_lstm_baseline():
    """Tests the LSTM baseline model.
    """
    d_model = 8
    dummy_vocab = torchtext.vocab.vocab({'1': 1, '2': 2})
    model = baselines.Baseline(d_model=d_model, vocab_size=len(dummy_vocab))
    dummy_inputs = torch.randint(0, len(dummy_vocab), (16, 32))
    outputs = model(dummy_inputs)
    assert outputs.shape == (16, 32, len(dummy_vocab))

    # Do a sanity check by making sure the model can overfit
    dummy_target = torch.randint(0, len(dummy_vocab), (16, 32))

    def loss_fn(inputs, target):
        print(inputs.shape, target.shape)
        inputs = inputs.reshape(-1, len(dummy_vocab))
        target = target.reshape(-1)
        print(inputs.shape, target.shape)
        return torch.nn.functional.cross_entropy(inputs, target)

    optimizer = torch.optim.Adam(model.parameters())
    losses = [loss_fn(outputs, dummy_target)]
    for _epoch in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = loss_fn(outputs, dummy_target)
        loss.backward()
        losses.append(loss.detach().cpu().item())
        optimizer.step()
    assert losses[0] > losses[-1]


def test_lstm_position_baseline():
    """Tests the LSTM position baseline model.
    """
    d_model = 8
    dummy_vocab = torchtext.vocab.vocab({'1': 1, '2': 2})
    model = baselines.PositionBaseline(d_model=d_model,
                                       vocab_size=len(dummy_vocab))
    dummy_inputs = torch.randn(16, 32, 17, 8, 8)
    outputs = model(dummy_inputs)
    assert outputs.shape == (16, 32, len(dummy_vocab))

    # Do a sanity check by making sure the model can overfit
    dummy_target = torch.randint(0, len(dummy_vocab), (16, 32))

    def loss_fn(inputs, target):
        print(inputs.shape, target.shape)
        inputs = inputs.reshape(-1, len(dummy_vocab))
        target = target.reshape(-1)
        print(inputs.shape, target.shape)
        return torch.nn.functional.cross_entropy(inputs, target)

    optimizer = torch.optim.Adam(model.parameters())
    losses = [loss_fn(outputs, dummy_target)]
    for _epoch in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = loss_fn(outputs, dummy_target)
        loss.backward()
        losses.append(loss.detach().cpu().item())
        optimizer.step()
    assert losses[0] > losses[-1]
