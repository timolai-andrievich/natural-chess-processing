"""Module with tests for `layers` module.

Run tests via `pytest src`
"""
import os

import torch

from . import training_loop
from .. import data
from .. import models


def test_get_model():
    """Tests the `get_model` function.
    """
    minimal_config = {
        'model': {
            'name': 'Baseline',
            'params': {
                'd_model': 16,
                'vocab_size': 12
            }
        }
    }
    model = training_loop.get_model(minimal_config, vocab_size=1)
    assert isinstance(model, models.Baseline)


def test_get_optimizer():
    """Tests the `get_optimizer` function.
    """
    minimal_config = {'optimizer': {'name': 'Adam', 'params': {}}}
    model = torch.nn.Linear(1, 1)
    optimizer = training_loop.get_optimizer(minimal_config, model)
    assert isinstance(optimizer, torch.optim.Adam)


def test_get_scheduler():
    """Tests the `get_scheduler` function.
    """
    minimal_config = {'scheduler': {'name': 'LinearLR', 'params': {}}}
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = training_loop.get_scheduler(minimal_config, optimizer)
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


def test_get_dataset():
    """Tests the `get_dataset` function.
    """
    temp_file = 'temp.txt'
    minimal_config = {'dataset': {'name': 'MoveDataset', 'file': temp_file}}
    vocab = data.build_vocab()
    try:
        with open(temp_file, 'w', encoding='utf-8') as file:
            file.write("e2e4 1-0\n")
        print(os.listdir())
        dataset = training_loop.get_dataset(minimal_config, vocab=vocab)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    assert isinstance(dataset, data.MoveDataset)


def test_training_loop():
    """Tests `TrainingLoop` class.
    """
    torch.manual_seed(42)
    games_text = """e2e4 1-0
    e2e4 1-0
    """
    temp_file = 'temp.txt'
    config = {
        'model': {
            'name': 'Baseline',
            'params': {
                'd_model': 16,
                'vocab_size': 1973
            }
        },
        'dataset': {
            'name': 'MoveDataset',
            'file': temp_file
        },
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 1e-1
            }
        },
        'scheduler': {
            'name': 'ConstantLR',
            'params': {
                'factor': 1
            }
        },
        'training': {
            'batch_size': 1,
            'epochs': 10,
            'val_split': 0.5,
            'sequence_length': 8
        }
    }

    # Set up the dataset file.
    try:
        with open(temp_file, 'w', encoding='utf-8') as file:
            file.write(games_text)
        loop = training_loop.TrainingLoop(config, device='cpu')
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Check class attributes
    assert len(loop._val_loader) == 1  # pylint:disable=protected-access
    assert len(loop._train_loader) == 1  # pylint:disable=protected-access
    assert isinstance(loop.get_model(), models.Baseline)

    class Counter:  # pylint:disable=too-few-public-methods
        """A simple class that implements a counter.
        """

        def __init__(self):
            """Initializes counter with value 0.
            """
            self.value = 0

        def increment(self, *_args):
            """Increments counter by one.
            """
            self.value += 1

    epoch_counter = Counter()
    batch_counter = Counter()

    # Check that model overfits on a small dataset
    initial_accuracy = loop.get_validation_metrics(quiet=True)['Accuracy']
    loop.run(quiet=True,
             batch_callback=batch_counter.increment,
             epoch_callback=epoch_counter.increment)
    final_accuracy = loop.get_validation_metrics(quiet=True)['Accuracy']
    assert final_accuracy > initial_accuracy
    assert batch_counter.value == len(
        loop._train_loader) * config['training']['epochs']  # pylint:disable=protected-access
    assert epoch_counter.value == config['training']['epochs']


def test_cuda_training_loop():
    """Tests `TrainingLoop` class on a CUDA device, if
    one is available.
    """
    if not torch.cuda.is_available():
        return
    torch.manual_seed(42)
    games_text = """e2e4 1-0
    e2e4 1-0
    """
    temp_file = 'temp.txt'
    config = {
        'model': {
            'name': 'Baseline',
            'params': {
                'd_model': 16,
                'vocab_size': 1973
            }
        },
        'dataset': {
            'name': 'MoveDataset',
            'file': temp_file
        },
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 1e-1
            }
        },
        'scheduler': {
            'name': 'ConstantLR',
            'params': {
                'factor': 1
            }
        },
        'training': {
            'batch_size': 32,
            'epochs': 10,
            'val_split': 0.5,
            'sequence_length': 8
        }
    }

    # Set up the dataset file.
    try:
        with open(temp_file, 'w', encoding='utf-8') as file:
            file.write(games_text)
        loop = training_loop.TrainingLoop(config, device='cuda')
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Check class attributes
    assert len(loop._val_loader) == 1  # pylint:disable=protected-access
    assert len(loop._train_loader) == 1  # pylint:disable=protected-access
    assert isinstance(loop.get_model(), models.Baseline)

    # Check that model overfits on a small dataset
    initial_accuracy = loop.get_validation_metrics(quiet=True)['Accuracy']
    loop.run(quiet=True, )
    final_accuracy = loop.get_validation_metrics(quiet=True)['Accuracy']
    assert final_accuracy > initial_accuracy
