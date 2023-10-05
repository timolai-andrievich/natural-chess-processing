"""Module that generates a vocabulary of uci moves.

Function `build_vocab` builds a vocabulary from all possible uci moves,
game results, and special symbols.

Usage:
    `vocab = build_vocab()`
"""
import itertools
from typing import List

import torchtext


def get_cell_name_from_coordinates(x: int, y: int) -> str:
    """Returns the name of the cell in the given coordinates.
    Coordinates are counted from the a1 cell, with a->h being the x axis,
    and 1->8 being the y axis. Axes are 0-indexed.

    Args:
        x (int): The x coordinate of the cell.
        y (int): The y coordinate of the cell.

    Returns:
        str: The name of the cell.
    """
    if not 0 <= x < 8 or not 0 <= y < 8:
        raise KeyError(f'Coordinates ({x}, {y}) are out of bounds')
    letters = 'abcdefgh'
    return f'{letters[x]}{y + 1:.0f}'


def get_pawn_promotion_moves(x: int, y: int) -> List[str]:
    """Returns the list of cells that are reachable after a pawn promotion move,
    with the promotion figure as the last symbol.

    Args:
        x (int): The starting x coordinate of the move.
        y (int): The starting x coordinate of the move.

    Returns:
        List[str]: The list of cells that are reachable after a pawn promotion move,
    with the promotion figure as the last symbol.
    """
    if y not in (1, 6):
        return []
    promotions = 'nbrq'
    moves_without_promotions = []
    if y == 1:
        target_y = 0
    else:
        target_y = 7
    for target_x in range(max(x - 1, 0), min(x + 2, 8)):
        moves_without_promotions.append(
            get_cell_name_from_coordinates(target_x, target_y))
    result = []
    for move in moves_without_promotions:
        for figure in promotions:
            result.append(f'{move}{figure}')
    return result


def get_bishop_moves(x: int, y: int) -> List[str]:
    """Returns the list of names of cells that are reachable after
    one bishop move from the starting cell. 

    Args:
        x (int): Starting x coordinate.
        y (int): Starting y coordinate.

    Returns:
        List[str]: Returns the list of names of cells that are reachable after
    one bishop move from the starting cell. 
    """
    result = []
    deltas = [i for i in range(-7, 8) if i != 0]
    for delta in deltas:
        if 0 <= x + delta < 8 and 0 <= y + delta < 8:
            result.append(get_cell_name_from_coordinates(x + delta, y + delta))
        if 0 <= x + delta < 8 and 0 <= y - delta < 8:
            result.append(get_cell_name_from_coordinates(x + delta, y - delta))
    return result


def get_rook_moves(x: int, y: int) -> List[str]:
    """Returns the list of names of cells that are reachable after
    one rook move from the starting cell. 

    Args:
        x (int): Starting x coordinate.
        y (int): Starting y coordinate.

    Returns:
        List[str]: Returns the list of names of cells that are reachable after
    one rook move from the starting cell. 
    """
    result = []
    for delta in range(-7, 0):
        if 0 <= x + delta < 8:
            result.append(get_cell_name_from_coordinates(x + delta, y))
        if 0 <= y + delta < 8:
            result.append(get_cell_name_from_coordinates(x, y + delta))
    for delta in range(1, 8):
        if 0 <= x + delta < 8:
            result.append(get_cell_name_from_coordinates(x + delta, y))
        if 0 <= y + delta < 8:
            result.append(get_cell_name_from_coordinates(x, y + delta))
    return result


def get_knight_moves(x: int, y: int) -> List[str]:
    """Returns the list of names of cells that are reachable after
    one knight move from the starting cell. 

    Args:
        x (int): Starting x coordinate.
        y (int): Starting y coordinate.

    Returns:
        List[str]: Returns the list of names of cells that are reachable after
    one knight move from the starting cell. 
    """
    deltas = []
    deltas += itertools.product((1, -1), (2, -2))
    deltas += itertools.product((2, -2), (1, -1))
    result = []
    for delta_x, delta_y in deltas:
        if not 0 <= x + delta_x < 8 or not 0 <= y + delta_y < 8:
            continue
        result.append(get_cell_name_from_coordinates(x + delta_x, y + delta_y))
    return result


def get_all_moves_from_coordinates(x: int, y: int) -> List[str]:
    """Returns the list of names of cells that are reachable after
    one move from the starting cell. 

    Args:
        x (int): Starting x coordinate.
        y (int): Starting y coordinate.

    Returns:
        List[str]: Returns the list of names of cells that are reachable after
    one move from the starting cell. 
    """
    start = get_cell_name_from_coordinates(x, y)
    result = []
    added = set()
    all_cells: List[str] = []
    all_cells += get_pawn_promotion_moves(x, y)
    all_cells += get_bishop_moves(x, y)
    all_cells += get_rook_moves(x, y)
    all_cells += get_knight_moves(x, y)
    for cell in all_cells:
        if cell is not None and cell not in added:
            added.add(cell)
            result.append(f'{start}{cell}')
    return result


def build_vocab(specials=('<SOS>', '<PAD>')) -> torchtext.vocab.Vocab:
    """Build vocabulary of uci move names, game results, and special symbols.

    Returns:
        torchtext.vocab.Vocab: _description_
    """
    vocab_dict = {}
    moves = []
    for x in range(8):
        for y in range(8):
            moves += get_all_moves_from_coordinates(x, y)
    moves.sort()
    results = ['1/2-1/2', '1-0', '0-1']
    tokens = moves + results
    for i, token in enumerate(tokens, 1):
        vocab_dict[token] = i
    return torchtext.vocab.vocab(vocab_dict,
                                 specials=specials,
                                 special_first=False)


# TODO Cover move generation functions with tests.