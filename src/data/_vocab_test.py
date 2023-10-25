"""Contains test functions for the `data.vocab` module.

Run tests via `pytest src`
"""
import pytest
import torchtext

from src.data import vocab


def test_cell_names():
    """Tests the `get_cell_name_from_coordinates` function.
    """
    assert vocab.get_cell_name_from_coordinates(0, 0) == 'a1'
    assert vocab.get_cell_name_from_coordinates(7, 7) == 'h8'
    with pytest.raises(KeyError):
        vocab.get_cell_name_from_coordinates(-1, -1)
    with pytest.raises(KeyError):
        vocab.get_cell_name_from_coordinates(8, 8)


def test_pawn_promotion_moves():
    """Tests `get_pawn_promotion_moves` function.
    """
    assert 'a1q' in vocab.get_pawn_promotion_moves(0, 1)
    assert 'a8q' not in vocab.get_pawn_promotion_moves(0, 1)
    assert len(vocab.get_pawn_promotion_moves(0, 1)) == 8
    assert len(vocab.get_pawn_promotion_moves(1, 1)) == 12
    assert len(vocab.get_pawn_promotion_moves(2, 2)) == 0


def test_bishop_moves():
    """Tests `get_bishop_moves` function.
    """
    assert ({'b2', 'c3', 'd4', 'e5', 'f6', 'g7',
             'h8'} == set(vocab.get_bishop_moves(0, 0)))


def test_rook_moves():
    """Tests `get_rook_moves` function.
    """
    for x in range(8):
        for y in range(8):
            assert len(vocab.get_rook_moves(x, y)) == 14
    assert 'a8' in vocab.get_rook_moves(0, 0)
    assert 'h1' in vocab.get_rook_moves(0, 0)


def test_knight_moves():
    """Tests `get_knight_moves` functions.
    """
    assert set({'b3', 'c2'}) == set(vocab.get_knight_moves(0, 0))


def test_build_vocab():
    """Tests `build_vocab` function.
    """
    vocabulary = vocab.build_vocab(specials=('<TEST>', '<PAD>', '<SOS>'))
    assert isinstance(vocabulary, torchtext.vocab.Vocab)
    assert '<TEST>' in vocabulary.get_stoi()
