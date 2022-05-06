"""
    Automated Supervised Training of a Model in Pytorch
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: September 2019 - October 2021
    License: MIT
"""

from __future__ import annotations
from typing import Union, Sequence, Mapping, Tuple, Callable, Any, MutableMapping, Optional
import gc
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from numbers import Real
from torch.optim import AdamW
from datetime import timedelta
from tqdm import tqdm
import logging

from optim.autosaver import Autosaver
from optim.split import train_test_split


class Trainer(Autosaver):
    """
    Utility for Supervised Automated Training of a Pytorch Model
    
    Notes
    -----
    
    Does the following:
    
    - Automatically trains a model.
    - Uses batch gradient descent (or stochastic gradient descent, depending on a `batch_size`).
    - Early-stops if model does not evolve for more than `patience` epochs.
    - Keeps best parameters, as measured on `test_dataset`.
    - Saves and restores state (snapshot) of model, optimizer and trainer itself.
    - Exports pure model, free from gradients, to reduce its size on a disk.
    
    Parameters
    ----------
    model : nn.Module
        A model to train.
    criterion : nn.Criterion
        A criterion to use when computing error from y (target value) and y_hat (predicted value).
        Default: `nn.MSELoss(reduction='mean')`
    optimizer : Optional[torch.optim.optimizer.Optimizer]
        Object that computes gradients and updates weights.
        If None - `AdamW(lr=1e-3)` is used.
        Default: None
    lr_scheduler : Optional[torch.optim.lr_scheduler]
        Object that computes optimal `learning_rate` from test_loss after each epoch.
        For instance: `torch.optim.lr_scheduler.ReduceLROnPlateau`.
        You should manually tie `lr_scheduler` to an `optimizer` object.
        As `Trainer` will only call `lr_scheduler.step(test_loss)` after each epoch.
    batch_size : int
        Specifies the number of samples in a batch to use.
    weight_decay : Optional[float]
        Specifies coefficient to perform L2 regularization of each step.
        Be careful, as optimizer may also perform additional weight_decay, depending on its arguments.
        Default: 1e-5
    shuffle : bool
        Shuffle train train_dataset after each epoch.
        Default: True
    early_stopping : bool
        Stop training if error for test train_dataset didn't decrease at least for `tolerance` value
        for more than `patience` epochs.
        Default: False
    test_fraction : float
        A float value inside interval (0, 1), not including the edges.
        This value is ignored if you call `fit` method with both train and test datasets.
        Otherwise, if you call fit with only one `dataset`, it is split into train and test parts according to this
        fraction.
        Default: 0.1
    max_epochs : int
        Stop training when `max_epoch` has passed.
        Default: 100
    patience : int
        For how many epochs to wait for model to improve performance (i.e. - decrease error on test train_dataset)
        before stopping the process.
        Default: 10
    tolerance : float
        Error on test train_dataset should decrease at least for `tolerance` value with each next epoch.
        Otherwise learning process will be stopped after `patience` epochs if `early_stopping` is True.
        Default: 1e-4
    repair_parameters : bool
        Sometimes during training some weights can become zero or infinitely large or NaN.
        Turn on this option to automatically detect and fix these weights.
        In this case they are replaced with some small random values.
        Default: True
    callbacks : Optional[Sequence[Callable[[Trainer], Any]]]
        When epoch ends you can specify a callback(Trainer)
        Default: None
    measure_performance : None or Callable[[Trainer], float]
        When epoch ends you can specify a callback(Trainer) to measure performance of a model.
        This performance will be saved in history and displayed on a chart.
    autosave_dir : Optional[str]
        Specify directory to auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: '.'
    autosave_prefix : Optional[str]
        Specify file name prefix for auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: None
    autosave_interval : Optional[Union[int, timedelta]]
        Specify interval for auto saving.
        When integer n value is specified: autosaving is performed after each n iterations.
        When timedelta t value is specified: autosaving is performed each t time interval.
        When None is specified: no autosaving is performed.
        Default: int(5)
    log : Union[logging.Logger, str, None]
        Specify logging.Logger, str or None.
        You may specify a logging object or a name of logging stream to receive
        some debug, information or warnings from Trainer.
        Default: None
    """
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module = nn.MSELoss(reduction='mean'),
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 lr_scheduler: Optional[object] = None,
                 batch_size: int = 200,
                 weight_decay: Optional[float] = 1e-5,
                 shuffle=True,
                 early_stopping=False,
                 test_fraction: float = 0.1,
                 max_epochs: int = 100,
                 patience: int = 10,
                 tolerance: float = 1e-4,
                 repair_parameters=True,
                 callbacks: Optional[Sequence[Callable[[Trainer], Any]]] = None,
                 measure_performance: Union[None, Callable[[Trainer], float]] = None,
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = 5,
                 log: Union[logging.Logger, str, None] = None):
        """
        
        """
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        # Save base objects
        self.model = model
        self.criterion = criterion
        parameters = model.parameters()
        if optimizer is not None:
            self.optimizer: Optional[torch.optim.Optimizer] = optimizer
        else:
            try:
                # Try to setup optimizer for model parameters
                self.optimizer: Optional[torch.optim.Optimizer] = AdamW(parameters, lr=1e-3)
            except ValueError:
                # Not all models require external optimizer!
                # Some models perform internal learning and do not manifest any parameters outside.
                self.optimizer: Optional[torch.optim.Optimizer] = None
        assert (lr_scheduler is None) or (isinstance(lr_scheduler, object) and hasattr(lr_scheduler, 'step'))
        self.lr_scheduler: Optional[object] = lr_scheduler  # torch.optim.lr_scheduler.ReduceLROnPlateau
        # Save arguments
        assert isinstance(batch_size, int) and (batch_size > 0)
        self.batch_size = batch_size
        assert (weight_decay is None) or (isinstance(weight_decay, float) and (weight_decay >= 0))
        self.weight_decay: Optional[float] = weight_decay
        self.shuffle = shuffle
        self.early_stopping = early_stopping
        assert isinstance(test_fraction, float) and (0 < test_fraction < 1)
        self.test_fraction = test_fraction
        assert isinstance(max_epochs, int) and (max_epochs > 0)
        self.max_epochs = max_epochs
        assert isinstance(patience, int) and (patience > 0)
        self.patience = patience
        assert isinstance(tolerance, float) and (tolerance > 0)
        self.tolerance = tolerance
        self.repair_parameters = repair_parameters
        assert (callbacks is None) or isinstance(callbacks, Sequence) or callable(callbacks)
        self.callbacks = [callbacks] if callable(callbacks) else callbacks
        assert (measure_performance is None) or callable(measure_performance)
        self.measure_performance = measure_performance
        
        if self.log is None:
            self.log: logging.Logger = logging.getLogger(model.__class__.__name__)

        # Initialize state variables
        self.n_epoch = 0
        self.n_iter = 0
        self.best_test_loss: Optional[Real] = None
        self.best_parameters: Optional[Mapping[str, Any]] = None
        self.no_improvement_count = 0
        self.train_loss_history = []
        self.test_loss_history = []
        self.performance_history = []
        
        # Try to load previously saved state
        self.autoload()

    def reset(self):
        """Resets state variables. Also calls model.reset() if it has such method"""
        self.n_epoch = 0
        self.n_iter = 0
        self.best_test_loss = None
        self.best_parameters = None
        self.no_improvement_count = 0
        self.train_loss_history = []
        self.test_loss_history = []
        self.performance_history = []
        if hasattr(self.model, 'reset') and callable(self.model.reset):
            self.model.reset()
            
    def fit(self, dataset: [Sequence], test_dataset: Optional[Sequence] = None):
        # Ensure we have got both train and test datasets
        if test_dataset is None:
            # Split dataset into train and test
            # train_indexes, test_indexes = Trainer.perform_stratified_split(y, self.test_fraction)
            train_indexes, test_indexes = train_test_split(len(dataset), self.test_fraction)
            test_dataset = dataset
        else:
            train_indexes = np.arange(len(dataset))
            test_indexes = np.arange(len(test_dataset))
        
        # Get number of train and test samples, batch_size
        n_train_samples, n_test_samples = len(train_indexes), len(test_indexes)
        batch_size = np.clip(self.batch_size, 1, min(n_train_samples, n_test_samples))
        n_train_batches = int(np.ceil(n_train_samples / batch_size))
        n_test_batches = int(np.ceil(n_test_samples / batch_size))

        if self.n_epoch >= self.max_epochs:
            self.log.info(f'Already reached max_epochs: {self.max_epochs}')
            return

        try:
            self.log.info(f'Starting training from epoch {self.n_epoch + 1} up to {self.max_epochs}')
            model_size = self.get_model_size(self.model)
            self.log.info(f'Model has {model_size[0]} total and {model_size[1]} trainable parameters')
            
            # Iterate over epochs
            while self.n_epoch < self.max_epochs:
                # Shuffle train indexes if needed
                if self.shuffle:
                    train_indexes = np.random.permutation(train_indexes)
                    
                # Reset train and test epoch indexes and registers
                train_index, test_index = 0, 0
                train_batch, test_batch = 0, 0
                accumulated_train_loss, accumulated_test_loss = 0, 0
    
                # Force clear unused memory
                gc.collect()
                
                # Iterate over batches in train and test datasets
                with tqdm(total=(n_train_batches + n_test_batches), ncols=80) as pbar:
                    while (train_index < n_train_samples) or (test_index < n_test_samples):
                        # Choose training or testing on this iteration
                        if (test_index / n_test_samples) < (train_index / n_train_samples):
                            # Perform testing:
                            
                            self.model.eval()
                            self.criterion.eval()
        
                            # Get test batch
                            x, y = test_dataset[test_indexes[test_index:test_index + batch_size]]
        
                            # Predict
                            y_hat = self.model.forward(x)
                            # Calculate overall test loss
                            loss = self.criterion(y_hat, y)
                            loss_scalar = loss.detach().item()
                            accumulated_test_loss += loss_scalar
        
                            # Increment test iteration counter
                            test_index = test_index + len(x)
                            test_batch = int(np.ceil(min(n_test_samples, (test_index - 1)) / batch_size))
        
                        else:
                            # Perform training:
                            self.model.train()
                            self.criterion.train()
    
                            # Get next batch inputs x and targets y
                            x, y = dataset[train_indexes[train_index:train_index + batch_size]]
                            
                            # Pass x through model and get predictions y_hat
                            y_hat = self.model.forward(x)
                            
                            # Calculate overall train loss
                            loss = self.criterion(y_hat, y)
                            loss_scalar = loss.detach().item()
                            accumulated_train_loss += loss_scalar
                            
                            # Update network weights
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
            
                            # Perform weight decay (L2 regularization) if needed
                            if isinstance(self.weight_decay, float) and (self.weight_decay > 0):
                                self.perform_weight_decay(self.optimizer.param_groups, self.weight_decay)
            
                            # Check and fix broken parameters if any
                            if self.repair_parameters:
                                self.perform_repair_parameters(self.optimizer.param_groups)
        
                            # Increment train iteration counter
                            train_index = train_index + len(x)
                            train_batch = int(np.ceil(min(n_train_samples, (train_index - 1)) / batch_size))
                            self.n_iter += 1
                            
                        # Print intermediate results
                        self.log.log(
                            logging.DEBUG - 1,
                            f'epoch {self.n_epoch} '
                            f'iter {train_index + test_index} '
                            f'loss {loss_scalar:.4f}' +
                            f'progress {(100*(train_batch + test_batch)/(n_train_batches + n_test_batches)):.2f}'
                        )
                        
                        pbar.update(train_batch + test_batch - pbar.n)
                    
                # Compute mean train and test loss for epoch
                train_loss = accumulated_train_loss * batch_size / n_train_samples
                test_loss = accumulated_test_loss * batch_size / n_test_samples
                self.train_loss_history.append(train_loss)
                self.test_loss_history.append(test_loss)
    
                # Increment epoch counter
                self.n_epoch += 1
    
                # Measure performance if needed
                performance = None
                if callable(self.measure_performance):
                    performance = self.measure_performance(self)
                    self.performance_history.append(performance)
                
                # Print epoch results
                self.log.info(
                    f'Epoch: {self.n_epoch}/{self.max_epochs}, '
                    f'iter: {self.n_iter}, '
                    f'lr: {self.learning_rate}, '
                    f'train: {train_loss:.4f}, '
                    f'test: {test_loss:.4f}' +
                    (f', perf: {performance:.2f}' if (performance is not None) else '')
                )
                
                # Update learning_rate if needed
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(test_loss)
                    
                # Check for new best result
                if (self.best_test_loss is None) or (self.best_test_loss > test_loss + self.tolerance):
                    # Save current best parameters
                    self.best_parameters = copy.deepcopy(self.model.state_dict())
                    self.no_improvement_count = 0
                    self.best_test_loss = test_loss
                else:
                    self.no_improvement_count += 1
    
                # Perform autosave if needed
                self.autosave(self.n_epoch, force=(self.no_improvement_count == 0))
    
                # Callback on_epoch_end
                if self.callbacks is not None:
                    for callback in self.callbacks:
                        if callable(callback):
                            callback(self)
                            
                # Check for early stopping
                if self.early_stopping and (self.no_improvement_count > self.patience):
                    self.log.info(
                        f'Test score did not improve more than tolerance={self.tolerance} '
                        f'for {self.patience} consecutive epochs. Stopping.'
                    )
                    break
    
            self.log.info('Finished training')
        
        except StopIteration:
            self.log.info('Training was stopped.')

        except KeyboardInterrupt:
            self.log.warning('Training was interrupted by user.')

        except InterruptedError:
            self.log.warning('Training was interrupted by system.')
        
        if self.early_stopping:
            # Check if model has converged or not
            if (self.n_epoch >= self.max_epochs) and (self.no_improvement_count > self.patience):
                # Print out warning
                self.log.warning(f'Stochastic Optimizer: Maximum epochs ({self.max_epochs}) '
                                 'reached and the optimization hasn\'t converged yet.')
            else:
                # Load best parameters
                self.log.info('Loading best parameters')
                self.model.load_state_dict(self.best_parameters, strict=True)

        # Forced autosave
        self.autosave(force=True)

    @staticmethod
    def get_model_size(model: nn.Module) -> Tuple[int, int]:
        total_parameters, trainable_parameters = 0, 0
        for p in model.parameters():
            n = p.numel()
            total_parameters += n
            if p.requires_grad:
                trainable_parameters += n
        return total_parameters, trainable_parameters
    
    @staticmethod
    def generate_batches(n, batch_size, min_batch_size=1):
        """
        Generator to create slices containing batch_size elements, from 0 to n.
        The last slice may contain less than batch_size elements, when batch_size
        does not divide n perfectly.
        """
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            if end + min_batch_size > n:
                continue
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)
    
    @staticmethod
    def perform_weight_decay(param_groups, weight_decay):
        if (weight_decay is None) or (weight_decay <= 0) or (weight_decay >= 1):
            return
        for group in param_groups:
            for param in group['params']:
                param.data = param.data.add(-weight_decay, param.data)

    def perform_repair_parameters(self, param_groups: Sequence[Mapping[str, Any]]):
        """Check and replace zero, NaN or inf parameters with random values"""
        for group in param_groups:
            for param in group['params']:
                if isinstance(param, torch.Tensor):
                    index = ((param.data != param.data) + (param.data == 0) +
                             (param.data == np.inf) + (param.data == -np.inf))
                    n = index.sum()
                    if n > 0:
                        if self.log:
                            self.log.warning(f'Repairing {n}/{param.numel()} bad parameters!')
                        param.data[index] = np.random.randn() / param.nelement()
                    index = ((param.data < -1e+10) + (param.data > 1e+10))
                    n = index.sum()
                    if n > 0:
                        if self.log:
                            self.log.warning(f'Clipping {n}/{param.numel()} huge parameters!')
                        param.data.clamp_(min=-1e+10, max=1e+10)

    def draw_chart(self, ax):
        import sys
        if 'matplotlib' not in sys.modules:
            # Turn off matplotlib debug messages
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if len(self.train_loss_history) < 2:
            return
        try:
            ax.set_xlabel('epoch')
            color_loss = 'tab:blue'
            ax.set_yscale('log')
            ax.set_ylabel('loss', color=color_loss)
            ax.plot(self.train_loss_history, label='train', color=color_loss)
            ax.plot(self.test_loss_history,  label='test', color='tab:green')
            ax.tick_params(axis='y', labelcolor=color_loss)
            ax.legend()
            if len(self.performance_history) > 0:
                color_perf = 'tab:red'
                ax2 = ax.twinx()
                ax2.set_ylabel('performance', color=color_perf)
                ax2.plot(self.performance_history, color=color_perf)
                ax2.tick_params(axis='y', labelcolor=color_perf)
        except Exception as ex:
            self.log.error(f'Failed to draw chart: {str(ex)}')
            pass
        
    def get_params(self, deep=True):
        return dict(
            criterion=self.criterion,
            lr_scheduler=self.lr_scheduler,
            batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            shuffle=self.shuffle,
            early_stopping=self.early_stopping,
            test_fraction=self.test_fraction,
            max_epochs=self.max_epochs,
            patience=self.patience,
            tolerance=self.tolerance,
            repair_parameters=self.repair_parameters,
            **super().get_params(deep)
        )
    
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict() if (self.optimizer is not None) else None,
            'lr_scheduler': self.lr_scheduler.state_dict() if (self.lr_scheduler is not None) else None,
            'n_iter': self.n_iter,
            'n_epoch': self.n_epoch,
            'best_test_loss': self.best_test_loss,
            'best_parameters': self.best_parameters,
            'no_improvement_count': self.no_improvement_count,
            'train_loss_history': self.train_loss_history,
            'test_loss_history': self.test_loss_history,
            'performance_history': self.performance_history,
        }
    
    def get_learning_rate(self):
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']
            break
        return learning_rate
        
    def set_learning_rate(self, learning_rate):
        self.log.info(f'Changing learning_rate to: {learning_rate}')
        for ids, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = learning_rate
    
    learning_rate = property(get_learning_rate, set_learning_rate)
    
    def load_state_dict(self, state, strict=False):
        if strict:
            required_keys = self.state_dict().keys()
            assert all(k in required_keys for k in state),\
                AssertionError(str(required_keys)+'\n'+str(state.keys()))
        if 'model' in state:
            self.model.load_state_dict(state['model'], strict=strict)
        if 'optimizer' in state:
            if (state['optimizer'] is not None) and (self.optimizer is not None):
                self.optimizer.load_state_dict(state['optimizer'])
        if 'lr_scheduler' in state:
            if (state['lr_scheduler'] is not None) and (self.lr_scheduler is not None):
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        # Use parent class method to load all other values
        super().load_state_dict(state=state, strict=False)
        
    def model_state(self):
        return self.model.state_dict()

    def load_model_state(self, model_state: Any, strict=False):
        self.model.load_state_dict(model_state, strict=strict)
