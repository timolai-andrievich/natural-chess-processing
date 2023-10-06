"""Contains test functions for the `data.dataset` module."""
from data.vocab import build_vocab
from data.dataset import MoveDataset


def test_move_dataset():
    """Tests the `MoveDataset` class.
    """
    dataset = MoveDataset(games=['e2e4', 'e2e4 e7e5', 'e2e4 e7e5 g1f3'])
    vocab = build_vocab()
    assert len(dataset) == 3
    assert dataset[0][0] == vocab.get_stoi()['e2e4']
    assert dataset[1][1] == vocab.get_stoi()['e7e5']
