"""Training script. Effectively provides a CLI for TrainingLoop class.
"""
import argparse
from typing import TypedDict, Optional
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from src import training_loop

# TODO add tensorboard logging
# TODO saving the model


class Args(TypedDict):
    """TypedDict containing command line arguments.
    """
    config_file_path: str
    device: str
    log_dir: str
    checkpoint_load: Optional[str]
    quiet: bool


def parse_args() -> Args:
    """
    Parses arguments and returns typed dict where defaults are inserted
    when applicable.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        dest='config_file_path',
                        required=True)
    parser.add_argument('-d',
                        '--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        dest='device',
                        required=True)
    parser.add_argument('--log-dir', type=str, dest='log_dir', required=True)
    parser.add_argument('--checkpoint', type=str, dest='log_dir', default=None)
    parser.add_argument('--quiet', action='store_true', dest='quiet')
    arguments = parser.parse_args()
    return arguments


def main():
    """The main function.
    """
    arguments = parse_args()
    config = training_loop.config.parse_config(arguments.config_file_path)
    loop = training_loop.TrainingLoop(config, arguments.device)
    loop.run(quiet=arguments.quiet)


if __name__ == '__main__':
    main()
