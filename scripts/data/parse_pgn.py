#!python
"""A script that parses chess game databased in .pgn format 
and outputs the moves from these games in uci format"""
import argparse
from dataclasses import dataclass
import io
import multiprocessing
import queue
import threading
import time
from typing import Optional

import chess.pgn
import tqdm

END_SIGNAL = None


@dataclass
class Args:
    """A class containing arguments passed to the program. Needed for type hints.
    """
    pgn_file: str
    output_file: str
    threads: int
    queue_size: int
    chunk_size: int
    quiet: bool


def parse_args() -> Args:
    """Parses arguments and returns a namespace with parsed arguments.

    Returns:
        Args: Arguments passed to the program.
    """
    parser = argparse.ArgumentParser("pgn2csv")
    parser.add_argument(
        "-p",
        "-i",
        "--pgn",
        dest="pgn_file",
        required=True,
        help="Input .pgn database.",
    )
    parser.add_argument("-o",
                        "--output",
                        default="output.txt",
                        dest="output_file",
                        help="Output file.")
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        default=14,
        type=int,
        help="The number of threads processing the games.",
    )
    parser.add_argument(
        "-q",
        "--queue-size",
        dest="queue_size",
        default=10000,
        type=int,
        help="The maximum size of queues in which the strings "
        "containing raw pgn strings and the sequences of "
        "UCI format moves are put. Setting this value too "
        "high might cause excessive memory use.",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        dest="chunk_size",
        default=16 * 2**10,
        type=int,
        help="The size of chunks the file is being read in.",
    )
    parser.add_argument(
        '--quiet',
        dest='quiet',
        action='store_true',
        help='Do not display progress bars or pring messages',
    )
    return parser.parse_args()


def tokens_from_game_string(game_string: str) -> str:
    """Parses a string containing a game in the PGN format,
    and returns moves in the UCI format separated by spaces.

    Args:
        game_string (str): String containing game in PGN format.

    Returns:
        str: Moves in the UCI format separated by spaces.
    """
    game_io = io.StringIO(game_string)
    game = chess.pgn.read_game(game_io)
    result_string = " ".join([move.uci() for move in game.mainline_moves()] +
                             [game.headers["Result"]])
    return result_string


def process_games(in_queue: multiprocessing.Queue,
                  out_queue: multiprocessing.Queue):
    """Takes strings containing games in PGN format from `in_queue`,
    and puts corresponding strings returned by `tokens_from_game_string` to `out_queue`.
    Exits when `END_SIGNAL` is recieved from the queue.

    Args:
        in_queue (multiprocessing.Queue): Queue containing strings with games in PGN format.
        out_queue (multiprocessing.Queue): Queue where processed games will go.
    """
    while True:
        game_string = in_queue.get()
        if game_string == END_SIGNAL:
            break
        tokens = tokens_from_game_string(game_string)
        out_queue.put(tokens)


def write_games(
    out_queue: multiprocessing.Queue,
    out_file: str,
    chunk_size: int,
    pbar: Optional[tqdm.tqdm] = None,
):
    """A function that writes games to the output file.
    Exits when `END_SIGNAL` is recieved from the queue.

    Args:
        out_queue (multiprocessing.Queue): The queue with lines to write to the file.
        out_file (str): The name of the output file.
        chunk_size (int): The size of write buffer.
        pbar (Optional[tqdm.tqdm], optional): The progress bar to update. Defaults to None.
    """
    with open(out_file, encoding="utf-8", mode="w",
              buffering=chunk_size) as file:
        while True:
            string = out_queue.get()
            if string == END_SIGNAL:
                break
            file.write(string)
            file.write("\n")
            if pbar is not None:
                pbar.update(1)


def read_games(
    in_queue: multiprocessing.Queue,
    in_file: str,
    chunk_size: int,
    pbar: Optional[tqdm.tqdm] = None,
):
    """Reads games in PGN format from the input file, and puts them into the queue.

    Args:
        in_queue (multiprocessing.Queue): The queue that the read games will go in.
        in_file (str): The path to the input file.
        chunk_size (int): The reading buffer size.
        pbar (Optional[tqdm.tqdm], optional): The progress bar to be updated. Defaults to None.
    """
    with open(in_file, encoding="utf-8", buffering=chunk_size) as file:
        for line in file:
            if line.startswith("1."):
                in_queue.put(line)
            if pbar is not None:
                pbar.update(1)
    pbar.refresh()


def update_info(
    pbar: tqdm.tqdm,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    stop_event: threading.Event,
):
    """Updates `pbar` with a general info about the state of the queues.
    Exits when the `stop_event` is set.

    Args:
        pbar (tqdm.tqdm): The progress bar. Does not actually display a bar, but rather two strings.
        in_queue (queue.Queue): The queue with read games.
        out_queue (queue.Queue): The queue with processed games.
        stop_event (threading.Event): The threading event to control the thread.
    """
    while not stop_event.is_set():
        pbar.set_description_str(
            f"Games in queue to be processed: {in_queue.qsize()}, "
            f"Games in queue to be written: {out_queue.qsize()}.",
            refresh=True,
        )
        time.sleep(1)


def main():  # pylint:disable=too-many-locals
    """Main function of the scripts. Handles multithreading and multiprocessing.
    """
    args = parse_args()
    in_queue = multiprocessing.Queue(maxsize=args.queue_size)
    out_queue = multiprocessing.Queue(maxsize=args.queue_size)
    lines_count = 0
    games_count = 0
    with tqdm.tqdm(position=0, desc="Lines read",
                   disable=args.quiet) as pbar_lines, tqdm.tqdm(
                       position=1, desc="Games found",
                       disable=args.quiet) as pbar_games, open(
                           args.pgn_file,
                           encoding="utf-8",
                           buffering=args.chunk_size) as file:
        for line in file:
            lines_count += 1
            if line[0] == "1":
                games_count += 1
                pbar_games.update(1)
            pbar_lines.update(1)
    if not args.quiet:
        print(f"Counted {lines_count} lines, {games_count} games.")
    pbar_lines_read = tqdm.tqdm(total=lines_count,
                                desc="Lines read",
                                position=0,
                                disable=args.quiet)
    pbar_lines_written = tqdm.tqdm(total=games_count,
                                   desc="Lines written",
                                   position=1,
                                   disable=args.quiet)
    pbar_info = tqdm.tqdm(bar_format="{desc}", position=2, disable=args.quiet)
    stop_event = threading.Event()
    reading_thread = threading.Thread(
        target=read_games,
        kwargs={
            "in_queue": in_queue,
            "in_file": args.pgn_file,
            "chunk_size": args.chunk_size,
            "pbar": pbar_lines_read,
        },
    )
    writing_thread = threading.Thread(
        target=write_games,
        kwargs={
            "out_queue": out_queue,
            "out_file": args.output_file,
            "chunk_size": args.chunk_size,
            "pbar": pbar_lines_written,
        },
    )
    info_thread = threading.Thread(
        target=update_info,
        kwargs={
            "pbar": pbar_info,
            "in_queue": in_queue,
            "out_queue": out_queue,
            "stop_event": stop_event,
        },
    )
    worker_processes = [
        multiprocessing.Process(
            target=process_games,
            kwargs={
                "in_queue": in_queue,
                "out_queue": out_queue,
            },
        ) for i in range(args.threads)
    ]

    # Start all the threads
    reading_thread.start()
    info_thread.start()
    for worker_process in worker_processes:
        worker_process.start()
    writing_thread.start()
    try:
        # Wait for threads to finish
        reading_thread.join()
        for process in worker_processes:
            # Each worker thread exits when it recieves END_SIGNAL.
            in_queue.put(END_SIGNAL)

        for process in worker_processes:
            process.join()

        out_queue.put(END_SIGNAL)
        writing_thread.join()
        stop_event.set()
        info_thread.join()
        pbar_lines_read.close()
        pbar_lines_written.close()
        pbar_info.close()
    except KeyboardInterrupt:
        print(
            "Keyboard Interrupt recieved, attempting to shut down gracefully")
        stop_event.set()


if __name__ == "__main__":
    main()

# TODO file name inference
