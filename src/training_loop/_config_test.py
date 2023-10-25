"""Module with tests for `layers` module.

Run tests via `pytest src`
"""
import os

import pytest

from . import config

EXAMPLE_CONFIG = {
    'model': {
        'name': 'Baseline',
        'params': {}
    },
    'dataset': {
        'name': 'MoveDataset',
        'file': 'data/small.txt'
    },
    'optimizer': {
        'name': 'Adam',
        'params': {}
    },
    'scheduler': {
        'name': 'CosineAnnealingLR',
        'params': {}
    },
    'training': {
        'batch_size': 32,
        'epochs': 1000,
        'val_split': 0.1,
        'sequence_length': 128
    }
}

EXAMPLE_CONFIG_TEXT = b"""
[model]
name="Baseline"

[dataset]
name="MoveDataset"
file="data/small.txt"

[optimizer]
name="Adam"

[scheduler]
name="CosineAnnealingLR"

[training]
batch_size=32
epochs=1000
val_split=0.1
sequence_length=128
"""


def test_validate_config():
    """Tests `validate_config` function.
    """
    config.validate_config(EXAMPLE_CONFIG)
    with pytest.raises(config.InvalidConfigError):
        config.validate_config({})


def test_insert_default_values():
    """Tests `insert_default_values` function.
    """
    inputed_config = config.insert_default_values(EXAMPLE_CONFIG)
    assert inputed_config['model']['params'] == {}
    assert inputed_config['optimizer']['params'] == {}
    assert inputed_config['scheduler']['params'] == {}


def test_parse_config():
    """Tests `parse_config` function.
    """
    temp_file_name = 'temp.toml'
    try:
        with open(temp_file_name, 'wb') as file:
            file.write(EXAMPLE_CONFIG_TEXT)
        cfg = config.parse_config(temp_file_name)
        assert cfg == EXAMPLE_CONFIG
    finally:
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)
