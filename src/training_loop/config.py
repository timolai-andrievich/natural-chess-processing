"""Contains functions related to config parsing and validation.
"""
import copy
import tomllib
from typing import Dict


class InvalidConfigError(Exception):
    """Contains message about why config is invalid.
    """

    def __init__(self, message: str):
        """Contains message about why config is invalid.

        Args:
            message (str): _description_
        """
        super().__init__()
        self.message = message


def validate_config(config: Dict):
    """Validates config and raises appropriate errors if config
    is invalid.

    Args:
        config (Dict): Config read from a file.

    Raises:
        InvalidConfigError: Error with information why the config is invalid.
    """
    def check_and_raise(condition, message):
        if not condition:
            raise InvalidConfigError(message)

    check_and_raise('model' in config, 'Model is not specified')
    check_and_raise('name' in config['model'], 'Model name is not specified')

    check_and_raise('dataset' in config, 'Dataset is not specified')
    check_and_raise('name' in config['dataset'],
                    'Dataset name is not specified')
    check_and_raise('file' in config['dataset'],
                    'Dataset file path is not specified')

    check_and_raise('optimizer' in config, 'Optimizer is not specified')
    check_and_raise('name' in config['optimizer'],
                    'Optimizer name is not specified.')

    check_and_raise('scheduler' in config, 'Scheduler is not specified')
    check_and_raise('name' in config['scheduler'],
                    'Scheduler name is not specified.')

    check_and_raise('training' in config,
                    'Training parameters are not specified')
    check_and_raise(
        'batch_size' in config['training'], 'Batch size is not specified')
    check_and_raise('epochs' in config['training'],
                    'Epoch count is not specified')
    check_and_raise('val_split' in config['training'],
                    'Validation set proportion is not specified')
    check_and_raise('sequence_length' in config['training'],
                    'Sequence length is not specified')


def insert_default_values(config: Dict) -> Dict:
    """Inserts default values where applicable and returns
    modified config.

    Args:
        config (Dict): Config to be modified.

    Returns:
        Dict: Modified config.
    """
    config = copy.deepcopy(config)
    if not 'params' in config['model']:
        config['model']['params'] = {}
    if not 'params' in config['optimizer']:
        config['optimizer']['params'] = {}
    if not 'params' in config['scheduler']:
        config['scheduler']['params'] = {}
    return config


def parse_config(file_name: str) -> Dict:
    """Parses config from a file, validates it, inserts default values
    where applicable, and returns the resulting config.

    Args:
        file_name (str): The path to the config file.

    Returns:
        Dict: Parsed config.
    """
    with open(file_name, 'rb') as file:
        config = tomllib.load(file)
    validate_config(config)
    return config
