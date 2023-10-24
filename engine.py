import io
from typing import List
from sys import stdin, stdout, stderr

import chess

ENGINE_NAME = 'TODO'  # TODO
ENGINE_AUTHORS = 'TODO'  # TODO


class NoSuchCommand(Exception):
    pass


class PositionEngine:

    def __init__(self):
        self.board = chess.Board()
        self._stop = False

    def go(self, tokens: List[str]):
        legal_moves = [move.uci() for move in self.board.legal_moves]
        if not legal_moves:
            legal_moves = ['(none)']
        move = legal_moves[0]
        stdout.write(f'info pv {move}\n')
        if 'infinite' not in tokens:
            stdout.write(f'bestmove {move}\n')
        stdout.flush()

    def position(self, tokens):
        match tokens:
            case 'startpos', :
                self.board = chess.Board()
            case 'fen', fen:
                self.board = chess.Board(fen=fen)
            case 'startpos', 'moves', *moves:
                self.board = chess.Board()
                for move in moves:
                    self.board.push_uci(move)
            case 'fen', fen, 'moves', *moves:
                self.board = chess.Board(fen=fen)
                for move in moves:
                    self.board.push_uci(move)
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
        file = open('log.txt', 'w')
        while not self._stop:
            line = stdin.readline()
            command = line.strip()
            tokens = list(filter(bool, command.split()))
            print('command', *tokens, file=file, flush=True)
            try:
                match tokens:
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
            except Exception:
                print('command rejected', *tokens, file=file, flush=True)
                stdout.write(f'No such command: {command}\n')
                stdout.flush()


def main():
    engine = PositionEngine()
    engine.mainloop()


if __name__ == '__main__':
    main()
