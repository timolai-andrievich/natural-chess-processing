"""Contains test functions for the `data.dataset` module."""
import chess
import numpy as np

from data.vocab import build_vocab
import data.dataset
from data.dataset import encode_game, encode_position, bitboard_to_2d_bitboard


def test_move_dataset():
    """Tests the `MoveDataset` class.
    """
    dataset = data.dataset.MoveDataset(
        games=['e2e4', 'e2e4 e7e5', 'e2e4 e7e5 g1f3'])
    vocab = build_vocab()
    assert len(dataset) == 3
    assert dataset[0][0] == vocab.get_stoi()['e2e4']
    assert dataset[1][1] == vocab.get_stoi()['e7e5']


def test_bitboard_conversion():
    """Tests `bitboard_to_2d_bitboard` function.
    """
    bitboard = chess.Bitboard()
    assert (bitboard_to_2d_bitboard(bitboard) == np.zeros((8, 8), bool)).all()

    board = chess.Board()
    bitboard = board.kings
    expected_keyboard = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
                                 dtype=bool)
    converted_bitboard = bitboard_to_2d_bitboard(bitboard)
    assert (converted_bitboard == expected_keyboard).all()

    board.push_uci('e2e4')
    bitboard = board.pawns
    expected_keyboard = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
                                 dtype=bool)
    converted_bitboard = bitboard_to_2d_bitboard(bitboard)
    assert (converted_bitboard == expected_keyboard).all()


def test_encode_position():
    """Tests `encode_position` function.
    """
    position = chess.Board()
    expected_kings = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
    ],
                              dtype=np.float32)
    encoded_position = data.dataset.encode_position(position)
    assert encoded_position.shape == (17, 8, 8)
    assert np.all(expected_kings == encoded_position[5] + encoded_position[11])

    position.push_uci('e2e4')
    encoded_position = data.dataset.encode_position(position)
    expected_black_pawns = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
                                    dtype=np.float32)
    assert np.all(expected_black_pawns == encoded_position[6])
    assert np.all(encoded_position[15] == 1)


def test_encode_game():
    """Tests `encode_game` function.
    """
    vocab = build_vocab()
    game = 'e2e4 e7e5 1/2-1/2'
    positions, moves = encode_game(game, vocab)
    assert moves == vocab(['e2e4', 'e7e5', '1/2-1/2'])
    board = chess.Board()
    assert np.all(positions[0] == encode_position(board))
    board.push_uci('e2e4')
    assert np.all(positions[1] == encode_position(board))


def test_position_dataset():
    """Tests `PositionDataset` class.
    """
    dataset = data.dataset.PosiitonDataset(
        games=['e2e4', 'e2e4 e7e5', 'e2e4 e7e5 g1f3'])
    vocab = build_vocab()
    assert len(dataset) == 3
    assert dataset[0][1][0] == vocab.get_stoi()['e2e4']
    assert dataset[0][0][0].shape == (17, 8, 8)
