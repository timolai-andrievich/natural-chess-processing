"""Module with training loop logic.
"""
from typing import Dict, Tuple, List, Optional

import torch
import tqdm
from torch import nn
import torch.utils.data
import torchmetrics

from .. import data
from .. import models


def get_model(config: Dict) -> nn.Module:
    """Returns the model with the name matching the one in config.

    Args:
        config (Dict): The config dictionary.

    Returns:
        nn.Module: Initialized model.
    """
    model_name = config['model']['name']
    model_params = config['model']['params']
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


def get_dataset(config: Dict) -> torch.utils.data.Dataset:
    """Reads data and returns dataset with it inside.

    Args:
        config (Dict): Config dictionary.

    Returns:
        Dataset: `torch` dataset in format specified in config.
    """
    dataset_name = config['dataset']['name']
    dataset_file = config['dataset']['file']
    dataset_class = data.__dict__[dataset_name]
    with open(dataset_file, 'r', encoding='utf-8') as file:
        games = [line.strip() for line in file]
    return dataset_class(games)


class TrainingLoop:
    """Trains the model according to parameters passed in config.
    """

    # TODO add validation
    # TODO add callbacks

    def __init__(self, config: Dict, device: Optional[str] = None):
        """Trains the model according to parameters passed in config.

        Args:
            config (Dict): Training config.
            device (Optional[str]): Device to run the model on. By default,
            choses CUDA if available, cpu otherwise.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        self._model: nn.Module = get_model(config).to(self._device)
        self._optimizer: torch.optim.Optimizer = get_optimizer(
            config, self._model)
        self._scheduler: torch.optim.lr_scheduler.LRScheduler = get_scheduler(
            config, self._optimizer)
        self._dataset: torch.utils.data.Dataset = get_dataset(config)
        self._config = config
        self._vocab = data.build_vocab()
        self._pad_index = self._vocab.get_stoi()['<PAD>']
        self._sos_index = self._vocab.get_stoi()['<SOS>']

        # Initialize dataset loaders
        batch_size = self._config['training']['batch_size']
        total_len = len(self._dataset)
        val_len = int(total_len * self._config['training']['val_split'])
        train_len = total_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            self._dataset, [train_len, val_len])
        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=self._collate_batch)
        self._val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=self._collate_batch)

    def _collate_batch(
            self, batch: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collates batch of moves, and returns padded tensors with move. 

        Args:
            batch (List[List[int]]): List of sequences of moves represented as indexes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensor of inputs, and tensor of targets.
        """
        inputs = []
        targets = []
        seq_len = self._config['training']['sequence_length']
        for game in batch:
            game = [self._sos_index] + game
            game = game[:seq_len + 1]
            game += [self._pad_index] * (seq_len - len(game))
            x, y = game[:-1], game[1:]
            inputs.append(x)
            targets.append(y)
        x = torch.tensor(x)
        y = torch.tensor(y)
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
                                         num_classes=len(self._vocab))
        total = 0
        accumulated_accuracy = 0
        self._model.eval()
        pbar = tqdm.tqdm(total=len(self._val_loader),
                         position=1,
                         disable=quiet,
                         desc='Calculating validation set metrics')
        for inputs, targets in self._val_loader:
            pred = self._model(inputs)
            total += len(inputs)
            accumulated_accuracy += len(inputs) * \
                accuracy(pred, targets).detach().cpu().item()
            pbar.update(1)
        pbar.close()
        return {'Accuracy': accumulated_accuracy / total}

    def run(self, quiet: bool = False):
        """Run the training loop.

        Args:
            quiet (bool, optional): Whether to show the progress bar or not. Defaults to False.
        """
        epochs = self._config['training']['epochs']
        pbar = tqdm.tqdm(total=epochs * len(self._train_loader),
                         position=0,
                         disable=quiet)
        pbar.set_description('Training loop')
        for epoch in range(epochs):
            self._model.train()
            training_losses = []
            for inputs, target in self._train_loader:
                minibatch_loss = self._train_step(inputs, target)
                training_losses.append(minibatch_loss)
                pbar.update(1)
            self._model.eval()
            metrics = self.get_validation_metrics(quiet)
            metrics.update({
                'Training loss':
                torch.mean(torch.tensor(training_losses)).item(),
                'Epoch':
                epoch
            })
            pbar.set_postfix(metrics)
        pbar.close()

    def get_model(self) -> nn.Module:
        """Returns the model.

        Returns:
            nn.Module: Model in the training loop.
        """
        return self._model
