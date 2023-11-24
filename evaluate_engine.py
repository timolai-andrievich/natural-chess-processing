import argparse
from typing import TypedDict

import chess
import chess.engine
import tqdm


class Args(TypedDict):
    """TypedDict containing command line arguments.
    """
    engine: str
    dataset: str
    quiet: bool


def parse_args() -> Args:
    """
    Parses arguments and returns typed dict where defaults are inserted
    when applicable.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('engine',
                        metavar='engine_executable',
                        type=str,
                        help='Path to the engine executable')
    parser.add_argument(
        'dataset',
        metavar='dataset_file',
        type=str,
        help='Path to the .txt file containing moves in UCI format.')
    parser.add_argument('--quiet', action='store_true', dest='quiet')
    arguments = parser.parse_args()
    return arguments


def evaluate_on_game(engine: chess.engine.SimpleEngine,
                     game: str) -> tuple[int, int]:
    terminals = {'1-0', '0-1', '1/2-1/2'}
    total = 0
    correct = 0
    moves = game.strip().split(' ')
    board = chess.Board()
    for move in moves:
        if move in terminals or not move:
            break
        engine_play = engine.play(board, chess.engine.Limit(time=0.01))
        engine_move = engine_play.move.uci()
        total += 1
        if engine_move == move:
            correct += 1
        board.push_uci(move)
    return total, correct


def main():
    args = parse_args()
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    total, correct = 0, 0
    with open(args.dataset) as file, tqdm.tqdm() as pbar:
        for game in file:
            game_total, game_correct = evaluate_on_game(engine, game)
            total += game_total
            correct += game_correct
            pbar.update(1)
    engine.quit()
    print(f'Accuracy: {correct / total * 100:.2f}')


if __name__ == '__main__':
    main()
