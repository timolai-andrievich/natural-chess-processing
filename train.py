"""Training script. Effectively provides a CLI for TrainingLoop class.
"""
import argparse
import os
from typing import TypedDict, Optional, Dict
import warnings

import torch
import torch.utils.tensorboard as torchboard

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from src import training_loop


class Args(TypedDict):
    """TypedDict containing command line arguments.
    """
    config_file_path: str
    device: str
    save_dir: str
    checkpoint_path: str
    quiet: bool


def parse_args() -> Args:
    """
    Parses arguments and returns typed dict where defaults are inserted
    when applicable.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('config_file_path',
                        metavar='config',
                        type=str,
                        help='Path to the config file')
    parser.add_argument('-d',
                        '--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        dest='device',
                        default='cpu',
                        help='Device to train the model on.')
    parser.add_argument('--log-dir',
                        type=str,
                        dest='log_dir',
                        default=None,
                        help='The log directory for tensorboard runs.')
    parser.add_argument('--output-dir',
                        type=str,
                        dest='save_dir',
                        help='Directory the checkpoints are saved into.',
                        default='.')
    parser.add_argument('--checkpoint-path',
                        type=str,
                        dest='checkpoint_path',
                        help='File contaning a PyTorch model checkpoint.',
                        default=None)
    parser.add_argument('--quiet', action='store_true', dest='quiet')
    arguments = parser.parse_args()
    return arguments


class ModelSaver:
    """A simple saver class with logging and saving functionality
    """

    def __init__(self,
                 loop: training_loop.TrainingLoop,
                 saving_dir: str,
                 log_dir: str = None):
        """Initializes a simple saver class with logging and saving
        functions.

        Args:
            loop (TrainingLoop): Training loop with the model inside.
            saving_dir (str): Directory the models are being saved to.
            log_dir (str, optional): Tensorboard logging directory, ./runs
            by default.
        """
        self._loop = loop
        self._best_accuracy = None
        self._save_dir = saving_dir
        self._writer = torchboard.SummaryWriter(log_dir=log_dir)

    def _save_model(self, file_name):
        """Saves model into a file `file_name`

        Args:
            file_name (str): The name of the model file.
        """
        torch.save(self._loop.get_model(), file_name)

    def save(self, epoch_metrics: Dict[str, float]):
        """Saves the model into `last.ckpt`, and into `best.ckpt`
        if accuracy metric is the best seen.

        Args:
            epoch_metrics (Dict[str, float]): Metrics dictionary.
        """
        self._save_model(f'{self._save_dir}/last.ckpt')
        self.log(epoch_metrics)
        accuracy = epoch_metrics['Accuracy']
        if self._best_accuracy is None or self._best_accuracy < accuracy:
            self._best_accuracy = accuracy
            self._save_model(f'{self._save_dir}/best.ckpt')

    def log(self, metrics: Dict[str, float]):
        """Logs metrics into the directory.

        Args:
            metrics (Dict[str, float]): Metrics dictionary.
        """
        for metric, value in metrics.items():
            if metric in {'Step', 'Epoch'}:
                continue
            self._writer.add_scalar(metric, value, metrics['Step'])
        self._writer.flush()


def main():
    """The main function.
    """
    arguments = parse_args()
    if not os.path.exists(arguments.save_dir):
        os.mkdir(arguments.save_dir)
    config = training_loop.config.parse_config(arguments.config_file_path)
    loop = training_loop.TrainingLoop(config, arguments.device)
    if arguments.checkpoint_path is not None:
        checkpoint = torch.load(arguments.checkpoint_path,
                                map_location=arguments.device)
        loop.set_model(checkpoint)
    saver = ModelSaver(loop, arguments.save_dir, log_dir=arguments.log_dir)
    loop.run(quiet=arguments.quiet,
             epoch_callback=saver.save,
             batch_callback=saver.log)


if __name__ == '__main__':
    main()
