"""Training script. Effectively provides a CLI for TrainingLoop class.
"""
import argparse
import os
from typing import TypedDict, Optional, Dict
import warnings

import torch
from torch import nn
import torch.utils.tensorboard as torchboard

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from src import training_loop

# TODO add tensorboard logging


class Args(TypedDict):
    """TypedDict containing command line arguments.
    """
    config_file_path: str
    device: str
    save_dir: str
    weights_path: Optional[str]
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
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        dest='config_file_path',
                        required=True)
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        dest='device',
        default='cpu',
    )
    parser.add_argument('--log-dir', type=str, dest='log_dir', default=None)
    parser.add_argument('--output-dir',
                        type=str,
                        dest='save_dir',
                        required=True)
    parser.add_argument('--weights-path',
                        type=str,
                        dest='weights_path',
                        default=None)
    parser.add_argument('--checkpoint-path',
                        type=str,
                        dest='checkpoint_path',
                        default=None)
    parser.add_argument('--quiet', action='store_true', dest='quiet')
    arguments = parser.parse_args()
    return arguments


class ModelSaver:

    def __init__(self,
                 loop: training_loop.TrainingLoop,
                 saving_dir: str,
                 log_dir: str = None):
        self._loop = loop
        self._best_accuracy = None
        self._save_dir = saving_dir
        self._writer = torchboard.SummaryWriter(log_dir=log_dir)

    def _save_model(self, file_name):
        torch.save(self._loop.get_model(), file_name)

    def save(self, epoch_metrics: Dict[str, float]):
        self._save_model(f'{self._save_dir}/last.ckpt')
        self.log(epoch_metrics)
        accuracy = epoch_metrics['Accuracy']
        if self._best_accuracy is None or self._best_accuracy < accuracy:
            self._best_accuracy = accuracy
            self._save_model(f'{self._save_dir}/best.ckpt')

    def log(self, metrics: Dict[str, float]):
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
    if arguments.weights_path is not None:
        state_dict = torch.load(arguments.weights_path)
        loop.load_state_dict(state_dict)
    elif arguments.checkpoint_path is not None:
        checkpoint = torch.load(arguments.checkpoint_path,
                                map_location=arguments.device)
        loop.set_model(checkpoint)
    saver = ModelSaver(loop, arguments.save_dir, log_dir=arguments.log_dir)
    loop.run(quiet=arguments.quiet,
             epoch_callback=saver.save,
             batch_callback=saver.log)


if __name__ == '__main__':
    main()
