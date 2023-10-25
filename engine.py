import io
from typing import List
from sys import stdin, stdout, stderr

import chess
import numpy as np
import torch

from src.data.dataset import encode_position
from src.data.vocab import build_vocab

ENGINE_NAME = 'TODO'  # TODO
ENGINE_AUTHORS = 'TODO'  # TODO

TERMINALS = {'1-0', '0-1', '1/2-1/2'}


class NoSuchCommand(Exception):
    pass


class PositionEngine:

    def __init__(self):
        self.board = chess.Board()
        self._stop = False
        self._model = None
        self._positions = [encode_position(self.board)]
        self._vocab = build_vocab()
        # TODO load when option is set
        self._load_model('models/position/last.ckpt')

    def _load_model(self, model_path: str):
        # TODO load weights instead of checkpoint
        self._model = torch.load(model_path, map_location='cpu')

    def _get_move(self):
        if self._model is None:
            return None, None
        positions = np.array(self._positions)
        game_tensor = torch.tensor(self._positions)
        game_tensor = game_tensor.view(1, *game_tensor.shape)
        outputs = self._model(game_tensor)
        move_probs = outputs[0, -1]
        raw_bestmove = self._vocab.lookup_token(move_probs.argmax())
        if raw_bestmove in TERMINALS:
            suggested_result = raw_bestmove
        else:
            suggested_result = None
        legal_moves = [move.uci() for move in self.board.legal_moves]
        if not legal_moves:
            return None, suggested_result
        legal_indicies = self._vocab(legal_moves)
        mask = torch.zeros_like(move_probs) - torch.inf
        mask[legal_indicies] = 0
        move_probs = move_probs + mask
        bestmove = self._vocab.lookup_token(move_probs.argmax())
        return bestmove, suggested_result

    def go(self, tokens: List[str]):
        move, suggested_result = self._get_move()
        if move is None:
            move = '(none)'
        stdout.write(f'info pv {move}\n')
        match suggested_result:
            case '1-0':
                stdout.write('info string white wins')
            case '0-1':
                stdout.write('info string black wins')
            case '1/2-1/2':
                stdout.write('info string draw suggested')
        if 'infinite' not in tokens:
            stdout.write(f'bestmove {move}\n')
        stdout.flush()

    def position(self, tokens):
        match tokens:
            case 'startpos', :
                self.board = chess.Board()
                self._positions = [encode_position(self.board)]
            case 'fen', fen:
                self.board = chess.Board(fen=fen)
            case 'startpos', 'moves', *moves:
                self.board = chess.Board()
                self._positions = [encode_position(self.board)]
                for move in moves:
                    self.board.push_uci(move)
                    self._positions.append(encode_position(self.board))
            case 'fen', fen, 'moves', *moves:
                self.board = chess.Board(fen=fen)
                self._positions = [encode_position(self.board)]
                for move in moves:
                    self.board.push_uci(move)
                    self._positions.append(encode_position(self.board))
            case _otherwise:
                raise NoSuchCommand()

    def uci(self):
        stdout.write(f'id name {ENGINE_NAME}\n')
        stdout.write(f'id author {ENGINE_AUTHORS}\n')
        stdout.write('uciok\n')
        stdout.flush()

    def isready(self):
        stdout.write('readyok\n')
        stdout.flush()

    def quit(self):
        self._stop = True

    def mainloop(self):
        while not self._stop:
            line = stdin.readline()
            command = line.strip()
            tokens = list(filter(bool, command.split()))
            try:
                match tokens:
                    case 'go', :
                        self.go([])
                    case 'go', *tokens:
                        self.go(tokens)
                    case 'uci', :
                        self.uci()
                    case 'isready', :
                        self.isready()
                    case 'quit', :
                        self.quit()
                    case 'position', *tokens:
                        self.position(tokens)
                    case 'setoption', *tokens:
                        pass
                    case 'stop', :
                        self.go([])
                    case 'ucinewgame', :
                        pass
                    case _otherwise:
                        raise NoSuchCommand()
            except Exception as e:
                stdout.write(f'No such command: {command}\n')
                stdout.flush()


def main():
    engine = PositionEngine()
    engine.mainloop()


if __name__ == '__main__':
    main()
