"""Contains dataset classes.
"""
from typing import List

import torch
import torch.utils.data

from data.vocab import build_vocab


class MoveDataset(torch.utils.data.Dataset):
    """A dataset of chess uci moves.
    
    Contains sequences of integers corresponding to moves.
    """

    def __init__(self, games: List[str]):
        """Initializes dataset from the passed games. Pads
        games to `max_len`, and appends a <SOS> token at the start.

        Args:
            games (List[str]): The list of strings, containing
            chess moves in uci format, separated by spaces.
        """
        self._games = []
        self._vocab = build_vocab()
        for game in games:
            moves = game.split(' ')
            self._games.append(self._vocab(moves))

    def __getitem__(self, index) -> List[int]:
        return self._games[index]

    def __len__(self):
        return len(self._games)

    @staticmethod
    def from_file(filename: str):
        """Reads lines from the file, and initializes the dataset with them.

        Args:
            filename (str): The name of the file contatining the games.

        Returns:
            MoveDataset: The initialized dataset.
        """
        with open(filename, 'r', encoding='utf-8') as file:
            return MoveDataset(games=file.readlines())


# TODO add position dataset
