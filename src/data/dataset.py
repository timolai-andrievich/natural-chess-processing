"""Contains dataset classes.
"""
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.utils.data
import torchtext


class MoveDataset(torch.utils.data.Dataset):
    """A dataset of chess uci moves.
    
    Contains sequences of integers corresponding to moves.
    """

    def __init__(self, games: List[str], vocab: torchtext.vocab.Vocab):
        """Initializes dataset from the passed games. Games are not
        padded and may have varying length.

        Args:
            games (List[str]): The list of strings, containing
            chess moves in uci format, separated by spaces.
            vocab (Vocab, optional): Vocabulary used to tokenize
            moves. If None, uses vocabulary returned by build_vocab
            function.
        """
        self._games = []
        self._vocab = vocab
        for game in games:
            if not game:
                continue
            moves = game.split(' ')
            self._games.append(self._vocab(moves))

    def __getitem__(self, index) -> List[int]:
        return self._games[index]

    def __len__(self):
        return len(self._games)


def bitboard_to_2d_bitboard(bitboard: chess.Bitboard) -> np.ndarray:
    """Coverts flat bitboard to numpy bool matrix.

    Args:
        bitboard (chess.Bitboard): Bitboard to be converted.

    Returns:
        np.ndarray: Boolean matrix corresponding to the given bitboard.
    """
    result = np.zeros((8, 8), dtype=bool)
    for square, bb_square in zip(chess.SQUARES, chess.BB_SQUARES):
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if bitboard & bb_square:
            result[rank][file] = 1
    return result


def encode_position(position: chess.Board) -> np.ndarray:
    """Encodes position using encoder similar to the encoder
    of Leela Chess Zero. Positions are encoded as a stack of
    17 8x8 bitboards with shape (channels, rows, columns).
    The bitboards are as follows (white and black do not
    denote actual colors, if it is black's move, the board
    is flipped):

    0. White pawns and en passant information
    1. White knights
    2. White bishops
    3. White rooks
    4. White queens
    5. White king

    6. Black pawns and en passant information
    7. Black knights
    8. Black bishops
    9. Black rooks
    10. Black queens
    11. Black king

    12. Rooks with castling rights
    13. 50-rule information
    14. Plane filled with ones to help NN find edges
    15. Plane filled with ones if the board was flipped
    16. Plane filled with ones if the game is over
    
    Args:
        position (chess.Board): Posiiton to be encoded.
    """
    # Flip the board if required
    result = np.zeros((17, 8, 8), np.float32)
    flipped = False
    if position.turn == chess.BLACK:
        position = position.mirror()
        flipped = True

    white_pieces = chess.Bitboard()
    for piece in chess.PIECE_TYPES:
        white_pieces |= position.pieces_mask(piece, chess.WHITE)
    black_pieces = chess.Bitboard()
    for piece in chess.PIECE_TYPES:
        black_pieces |= position.pieces_mask(piece, chess.BLACK)

    # Encode white pieces information
    result[0, :, :] = bitboard_to_2d_bitboard(white_pieces & position.pawns)
    result[1, :, :] = bitboard_to_2d_bitboard(white_pieces & position.knights)
    result[2, :, :] = bitboard_to_2d_bitboard(white_pieces & position.bishops)
    result[3, :, :] = bitboard_to_2d_bitboard(white_pieces & position.rooks)
    result[4, :, :] = bitboard_to_2d_bitboard(white_pieces & position.queens)
    result[5, :, :] = bitboard_to_2d_bitboard(white_pieces & position.kings)

    # Encode black pieces information
    result[6, :, :] = bitboard_to_2d_bitboard(black_pieces & position.pawns)
    result[7, :, :] = bitboard_to_2d_bitboard(black_pieces & position.knights)
    result[8, :, :] = bitboard_to_2d_bitboard(black_pieces & position.bishops)
    result[9, :, :] = bitboard_to_2d_bitboard(black_pieces & position.rooks)
    result[10, :, :] = bitboard_to_2d_bitboard(black_pieces & position.queens)
    result[11, :, :] = bitboard_to_2d_bitboard(black_pieces & position.kings)

    # Encode en passant information
    if position.has_legal_en_passant():
        en_passant_square = position.ep_square
        en_passant_file = chess.square_file(en_passant_square)
        result[0, 7, en_passant_file] = 1
        result[6, 7, en_passant_file] = 1

    result[12, :, :] = bitboard_to_2d_bitboard(position.castling_rights)
    result[13, :, :] = position.halfmove_clock / 100
    result[14, :, :] = 1
    if flipped:
        result[15, :, :] = 1
    if position.is_game_over():
        result[16, :, :] = 1
    return result


def encode_game(
        game: str,
        vocab: torchtext.vocab.Vocab) -> Tuple[List[np.ndarray], List[int]]:
    """Encodes a game, and returns a list of encoded positions
    and a list move tokens.

    Args:
        game (str): The string containing moves in uci format, separated
        by spaces.

    Returns:
        Tuple[List[np.ndarray], List[int]]: List of positions and list of
        moves.
    """
    token_map = vocab.get_stoi()
    encoded_positions = []
    encoded_moves = []
    tokens = game.split(' ')
    board = chess.Board()
    for token in tokens:
        encoded_position = encode_position(board)
        encoded_move = token_map[token]

        encoded_positions.append(encoded_position)
        encoded_moves.append(encoded_move)

        if token in {'1-0', '0-1', '1/2-1/2'}:
            assert token == tokens[-1] and tokens.count(token) == 1
            break

        board.push(chess.Move.from_uci(token))
    return encoded_positions, encoded_moves


class PositionDataset(torch.utils.data.Dataset):
    """A dataset of chess position and moves made from them.
    
    Contains encoded positions and corresponding moves.
    Positions are encoded similar to how Leela Chess Zero chess
    engine encodes them: as a stack of bitboards. See source
    code for details.
    """

    def __init__(self, games: List[str], vocab: torchtext.vocab.Vocab):
        """Initializes dataset from the passed games. Games are not
        padded and may have varying length. When indexed with index `i`,
        returns tuple of list encoded positions and list of encoded
        moves of the `i`-th game.

        Args:
            games (List[str]): The list of strings, containing
            chess moves in uci format, separated by spaces.
            vocab (Vocab): Vocabulary used to tokenize moves.
        """
        self._vocab = vocab
        self._games_positions = []
        self._games_moves = []
        for game in games:
            if not game:
                continue
            positions, moves = encode_game(game, self._vocab)
            self._games_positions.append(positions)
            self._games_moves.append(moves)

    def __getitem__(self, index):
        return self._games_positions[index], self._games_moves[index]

    def __len__(self):
        return len(self._games_positions)
