import argparse
import io
import multiprocessing
import queue
import threading
import time
from typing import Optional

import chess.pgn
import tqdm


def parse_args():
    parser = argparse.ArgumentParser("pgn2csv")
    parser.add_argument(
        "-p",
        "-i",
        "--pgn",
        dest="pgn_file",
        default="tiny_db.pgn",
        help="Input .pgn database.",
    )
    parser.add_argument(
        "-o", "--output", default="output.txt", dest="output_file", help="Output file."
    )
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
    return parser.parse_args()


def tokens_from_game_string(game_string: str) -> str:
    game_io = io.StringIO(game_string)
    game = chess.pgn.read_game(game_io)
    result_string = " ".join(
        [move.uci() for move in game.mainline_moves()] + [game.headers["Result"]])
    return result_string


def process_games(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue):
    while True:
        game_string = in_queue.get()
        if game_string is None:
            break
        tokens = tokens_from_game_string(game_string)
        out_queue.put(tokens)


def write_games(
    out_queue: multiprocessing.Queue,
    out_file: str,
    chunk_size: int,
    pbar: Optional[tqdm.tqdm] = None,
):
    with open(out_file, encoding="utf-8", mode="w", buffering=chunk_size) as file:
        while True:
            string = out_queue.get()
            if string is None:
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
    while not stop_event.is_set():
        pbar.set_description_str(
            f"Games in queue to be processed: {in_queue.qsize()}, Games in queue to be written: {out_queue.qsize()}.",
            refresh=True,
        )
        time.sleep(1)


def main():
    args = parse_args()
    input_file = args.pgn_file
    output_file = args.output_file
    threads = args.threads
    queue_maxsize = args.queue_size
    chunk_size = args.chunk_size
    in_queue = multiprocessing.Queue(maxsize=queue_maxsize)
    out_queue = multiprocessing.Queue(maxsize=queue_maxsize)
    lines_count = 0
    games_count = 0
    with tqdm.tqdm(position=0, desc="Lines read") as pbar_lines, tqdm.tqdm(
        position=1, desc="Games found"
    ) as pbar_games, open(input_file, encoding="utf-8", buffering=chunk_size) as file:
        for line in file:
            lines_count += 1
            if line[0] == "1":
                games_count += 1
                pbar_games.update(1)
            pbar_lines.update(1)
    print(f"Counted {lines_count} lines, {games_count} games.")
    pbar_lines_read = tqdm.tqdm(
        total=lines_count, desc="Lines read", position=0)
    pbar_lines_written = tqdm.tqdm(
        total=games_count, desc="Lines written", position=1)
    pbar_info = tqdm.tqdm(bar_format="{desc}", position=2)
    stop_event = threading.Event()
    reading_process = multiprocessing.Process(
        target=read_games,
        kwargs={
            "in_queue": in_queue,
            "in_file": input_file,
            "chunk_size": chunk_size,
            "pbar": pbar_lines_read,
        },
    )
    writing_process = multiprocessing.Process(
        target=write_games,
        kwargs={
            "out_queue": out_queue,
            "out_file": output_file,
            "chunk_size": chunk_size,
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
        )
        for i in range(threads)
    ]

    # Start all the threads
    reading_process.start()
    info_thread.start()
    for worker_process in worker_processes:
        worker_process.start()
    writing_process.start()
    try:
        # Wait for threads to finish
        reading_process.join()
        pbar_lines_read.refresh()
        pbar_lines_read.close()
        for process in worker_processes:
            # Each worker thread exits when it recieves None.
            in_queue.put(None)
            in_queue.put(None)
            # TODO figure out a real solution to process stopping

        for process in worker_processes:
            process.join()

        out_queue.put(None)
        out_queue.put(None)
        writing_process.join()
        pbar_lines_written.close()
        pbar_lines_written.refresh()
        stop_event.set()
        info_thread.join()
        pbar_info.refresh()
        pbar_info.close()
    except KeyboardInterrupt:
        print("Keyboard Interrupt recieved, attempting to shut down gracefully")
        stop_event.set()


if __name__ == "__main__":
    main()
