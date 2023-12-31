"""Module with training loop logic.
"""
import os
from typing import Dict, Tuple, List, Optional, Callable

import numpy as np
import torch
import tqdm
from torch import nn
import torch.utils.data
import torchmetrics
import torchtext

from .. import data
from .. import models


def get_model(config: Dict, vocab_size: int) -> nn.Module:
    """Returns the model with the name matching the one in config.

    Args:
        config (Dict): The config dictionary.
        vocab_size (Vocab): The size of vocabulary used in training loop.

    Returns:
        nn.Module: Initialized model.
    """
    model_name = config['model']['name']
    model_params = config['model']['params']
    if 'vocab_size' not in model_params:
        model_params['vocab_size'] = vocab_size
    model_class = models.__dict__[model_name]
    return model_class(**model_params)


def get_optimizer(config: Dict, model: nn.Module) -> torch.optim.Optimizer:
    """Returns the optimizer specified in config.

    Args:
        config (Dict): Config dictionary.
        model (nn.Module): The model which parameters need to be optimized.

    Returns:
        Optimizer: `torch` optimizer.
    """
    optimizer_name = config['optimizer']['name']
    optimizer_params = config['optimizer']['params']
    optimizer_class = torch.optim.__dict__[optimizer_name]
    return optimizer_class(params=model.parameters(), **optimizer_params)


def get_scheduler(
        config: Dict, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler:
    """Returns the scheduler specified in config.

    Args:
        config (Dict): Config dictionary.
        optimizer (nn.Module): The optimizer to which the scheduler will be applied.

    Returns:
        Optimizer: `torch` optimizer.
    """
    scheduler_name = config['scheduler']['name']
    scheduler_params = config['scheduler']['params']
    scheduler_class = torch.optim.lr_scheduler.__dict__[scheduler_name]
    return scheduler_class(optimizer=optimizer, **scheduler_params)


def get_dataset(config: Dict,
                vocab: torchtext.vocab.Vocab) -> torch.utils.data.Dataset:
    """Reads data and returns dataset with it inside.

    Args:
        config (Dict): Config dictionary.
        vocab (Vocab): Vocabulary used to tokenize moves.

    Returns:
        Dataset: `torch` dataset in format specified in config.
    """
    dataset_name = config['dataset']['name']
    dataset_file = config['dataset']['file']
    dataset_class = data.__dict__[dataset_name]
    with open(dataset_file, 'r', encoding='utf-8') as file:
        games = [line.strip() for line in file]
    return dataset_class(games, vocab=vocab)


class TrainingLoop:  # pylint: disable=too-many-instance-attributes
    """Trains the model according to parameters passed in config.
    """

    def __init__(self,
                 config: Dict,
                 device: Optional[str] = None,
                 num_workers: int = None):
        """Trains the model according to parameters passed in config.

        Args:
            config (Dict): Training config.
            device (Optional[str]): Device to run the model on. By default,
            choses CUDA if available, cpu otherwise.
            num_workers (int): Number of workers to use for loading data.
            If 0, data is loaded in the main thread. Defaults to 4 for
            posix systems, 0 for others. Values greater than 0 may not
            work for non-linux systems.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        self._vocab = data.build_vocab()
        self._model: nn.Module = get_model(config,
                                           len(self._vocab)).to(self._device)
        self._optimizer: torch.optim.Optimizer = get_optimizer(
            config, self._model)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
            config, self._optimizer)
        dataset: torch.utils.data.Dataset = get_dataset(config, self._vocab)
        self._config = config
        self._pad_index = self._vocab.get_stoi()['<PAD>']
        self._sos_index = self._vocab.get_stoi()['<SOS>']
        self._encode_positions = False
        if 'encode_positions' in self._config['training']:
            self._encode_positions = self._config['training'][
                'encode_positions']
        # Initialize dataset loaders
        batch_size = self._config['training']['batch_size']
        total_len = len(dataset)
        val_len = int(total_len * self._config['training']['val_split'])
        train_len = total_len - val_len
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_len, val_len])
        self.batch_size = batch_size
        if num_workers is None:
            num_workers = 4 if os.name == 'posix' else 0
        self._train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=num_workers)
        self._val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_batch,
            num_workers=num_workers)
        self._validation_pbar = None

    def _collate_batch(
            self, batch: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collates batch of moves, and returns padded tensors with move.

        Args:
            batch (List[List[int]]): List of sequences of
            moves represented as indexes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensor of inputs,
            and tensor of targets.
        """
        inputs = []
        targets = []
        seq_len = self._config['training']['sequence_length']
        if not self._encode_positions:
            for game in batch:
                game = [self._sos_index] + game
                game = game[:seq_len + 1]
                game += [self._pad_index] * (seq_len - len(game) + 1)
                input_, target = game[:-1], game[1:]
                inputs.append(input_)
                targets.append(target)
        else:
            pad_position = np.zeros_like(batch[0][0][0])
            for positions, moves in batch:
                positions = positions[:seq_len]
                positions += [pad_position] * (seq_len - len(positions))
                moves = moves[:seq_len]
                moves += [self._pad_index] * (seq_len - len(moves))
                inputs.append(positions)
                targets.append(moves)
            inputs = np.array(inputs)
        x = torch.tensor(inputs)
        y = torch.tensor(targets)
        return x, y

    def _loss_fn(self, inputs: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """Calculates cross-entropy between the inputs and the target while ignoring
        the padding.

        Args:
            inputs (torch.Tensor): Prediction logits.
            target (torch.Tensor): Target class indexes.

        Returns:
            torch.Tensor: Mean crossentropy over batch.
        """
        inputs = torch.flatten(inputs, end_dim=-2)
        target = torch.flatten(target)
        return nn.functional.cross_entropy(inputs,
                                           target,
                                           ignore_index=self._pad_index)

    def _train_step(self, inputs: torch.Tensor, target: torch.Tensor) -> float:
        """Makes a train step over the minibatch.

        Args:
            inputs (torch.Tensor): Minibatch inputs.
            target (torch.Tensor): Minibatch target.

        Returns:
            float: Minibatch loss.
        """
        inputs = inputs.to(self._device)
        target = target.to(self._device)
        pred = self._model(inputs)
        loss = self._loss_fn(pred, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._scheduler.step()
        return loss.detach().cpu().item()

    def get_validation_metrics(self, quiet: bool) -> Dict[str, float]:
        """Calculates metrics over validation set.

        Args:
            quiet (bool): Whether to show progress bar or not.

        Returns:
            Dict[str, float]: Dictionary mapping name of the metric to
            the value of the metric.
        """
        accuracy = torchmetrics.Accuracy(task='MULTICLASS',
                                         ignore_index=self._pad_index,
                                         num_classes=len(self._vocab)).to(
                                             self._device)
        total = 0
        accumulated_accuracy = 0
        accumulated_loss = 0
        self._model.eval()
        if self._validation_pbar is None:
            self._validation_pbar = tqdm.tqdm(
                total=len(self._val_loader),
                position=1,
                disable=quiet,
                desc='Calculating validation set metrics')
        else:
            self._validation_pbar.reset()
        for inputs, targets in self._val_loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            pred = self._model(inputs)
            pred = torch.flatten(pred, end_dim=-2)
            targets = torch.flatten(targets)
            total += len(inputs)
            accumulated_accuracy += len(inputs) * accuracy(  # pylint:disable=not-callable
                pred, targets).detach().cpu().item()
            accumulated_loss += self._loss_fn(pred,
                                              targets).item() * len(inputs)
            self._validation_pbar.update(1)
        self._validation_pbar.refresh()
        return {
            'Accuracy': accumulated_accuracy / total,
            'Validation loss': accumulated_loss / total
        }

    def run(self,
            quiet: bool = False,
            *,
            batch_callback: Optional[Callable[[], None]] = None,
            epoch_callback: Optional[Callable[[], None]] = None):
        """Run the training loop.

        Args:
            quiet (bool, optional): Whether to show the progress bar or not.
            Defaults to False.
            batch_callback (Callable[[Dict[str, float]], None], optional):
            Function that will be called after processing a batch.
            epoch_callback (Callable[[Dict[str, float]], None], optional):
            Function that will be called after every epoch.
        """
        if batch_callback is None:

            def batch_callback(*_args):
                pass

        if epoch_callback is None:

            def epoch_callback(*_args):
                pass

        epochs = self._config['training']['epochs']
        pbar = tqdm.tqdm(total=epochs * len(self._train_loader),
                         position=0,
                         disable=quiet)
        pbar.set_description('Training loop')
        step = 0
        for epoch in range(epochs):
            self._model.train()
            training_losses = []
            for inputs, target in self._train_loader:
                minibatch_loss = self._train_step(inputs, target)
                training_losses.append(minibatch_loss)
                pbar.update(1)
                training_metrics = {
                    'Epoch': epoch,
                    'Step': step,
                    'Batch loss': minibatch_loss
                }
                batch_callback(training_metrics)
                step += 1
            self._model.eval()
            metrics = self.get_validation_metrics(quiet)
            metrics.update({
                'Training loss':
                torch.mean(torch.tensor(training_losses)).item(),
                'Epoch':
                epoch,
                'Step':
                step,
            })
            epoch_callback(metrics)
            pbar.set_postfix(metrics)
        pbar.close()

    def get_model(self) -> nn.Module:
        """Returns the model.

        Returns:
            nn.Module: Model in the training loop.
        """
        return self._model

    def load_state_dict(self, state_dict: Dict):
        """Loads the provided state dict into the model.

        Args:
            state_dict (Dict): State dict to be loaded.
        """
        self._model.load_state_dict(state_dict)

    def set_model(self, model: nn.Module):
        """Sets the internal model variable. The model is not
        cloned, but the reference to the provided module is set instead

        Args:
            model (Module): PyTorch model.
        """
        self._model = model
