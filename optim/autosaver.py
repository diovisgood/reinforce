"""
    Implementation of Autosaver module for automatic saving and restoring checkpoints
    when training PyTorch models with different machine learning algorithms
    (supervised, unsupervised, Q-Learning, Policy Gradients, CMA-ES, etc.)
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: June 2021
    License: MIT
"""

from typing import Union, Mapping, Any, Optional
from datetime import timedelta, datetime
import logging
import torch
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Autosaver:
    """
    Base for autoload and autosave operations for classes with internal state
    
    Parameters
    ----------
    autosave_dir : Optional[str]
        Specify directory to auto save checkpoints.
        When None is specified - no autosaving is performed.
        Default: '.'
    autosave_prefix : Optional[str]
        Specify file name prefix to be prepended to saved state, model, chart and conf files.
        Default: None
    autosave_interval : Union[int, timedelta, None]
        Specify interval for auto saving.
        When integer n value is specified: autosaving is performed after each n iterations.
        When timedelta t value is specified: autosaving is performed each t time interval.
        When None is specified: no autosaving is performed.
        Default: int(5)
    log : Union[logging.Logger, str, None]
        Specify logging.Logger, str or None.
        You may specify a logging object or a name of logging stream to receive
        some debug, information or warning messages from Autosaver instance.
        Default: None
    """
    def __init__(self,
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = 5,
                 log: Union[logging.Logger, str, None] = None):
        assert ((autosave_dir is None) or
                (isinstance(autosave_dir, str) and os.path.isdir(autosave_dir)))
        self.autosave_dir: Optional[str] = autosave_dir

        assert ((autosave_prefix is None) or
                (isinstance(autosave_prefix, str) and set(autosave_prefix).isdisjoint({'/', '\\', os.path.sep})))
        if autosave_prefix is None:
            autosave_prefix = ''
        else:
            autosave_prefix += '.'
        self.autosave_prefix: str = autosave_prefix

        assert ((autosave_interval is None) or
                (isinstance(autosave_interval, int) and autosave_interval > 0) or
                (isinstance(autosave_interval, timedelta) and autosave_interval.total_seconds() > 0))
        self.autosave_interval: Optional[Union[int, timedelta]] = autosave_interval
        
        assert (log is None) or isinstance(log, (str, logging.Logger))
        if isinstance(log, str):
            log = logging.getLogger(log)
        self.log: Optional[logging.Logger] = log

        self._saved_config = False

        self.last_save_time: Optional[datetime] = None
        
    def _autosave_enabled(self):
        return (isinstance(self.autosave_dir, str) and
                isinstance(self.autosave_interval, (int, timedelta)))
    
    def autoload(self) -> bool:
        # Check if autosave is enabled
        if not self._autosave_enabled():
            return False
        # Try to load previously saved state
        state_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'state.pt')
        if (self.autosave_dir is not None) and (self.autosave_prefix is not None):
            if os.path.exists(state_file_name) and os.path.isfile(state_file_name):
                self.load_state(state_file_name, strict=True)
                if self.log:
                    self.log.info(f'Loaded state from {state_file_name}')
                self.last_save_time = datetime.now()
                return True
            else:
                if self.log:
                    self.log.info(f'Did not find previously saved state {state_file_name}')
                # Try to load previously saved model
                model_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'model.pt')
                if os.path.exists(model_file_name) and os.path.isfile(model_file_name):
                    try:
                        self.load_model(model_file_name, strict=True)
                        if self.log:
                            self.log.info(f'Loaded model from {model_file_name}')
                        self.last_save_time = datetime.now()
                        return True
                    except NotImplementedError:
                        pass
                else:
                    if self.log:
                        self.log.info(f'Did not find previously saved model {model_file_name}')
                return False
    
    def time_to_save(self, i: Optional[int] = None):
        if self.last_save_time is None:
            self.last_save_time = datetime.now()
            return False
        if (
            (isinstance(self.autosave_interval, int) and isinstance(i, int) and (i % self.autosave_interval == 0))
            or
            (
                isinstance(self.autosave_interval, timedelta) and
                (datetime.now() - self.last_save_time >= self.autosave_interval)
            )
        ):
            return True
        return False
    
    def autosave(self, i: Optional[int] = None, force=False):
        """
        Performs autosaving if it is time or when forced to.
        
        Notes
        -----
        A subclass should call this method after each iteration providing a call with iteration number.
        The method will automatically decide if it is time to perform autosave or not.
        A user can also call this method directly without iteration number but with specifying force=True.
        
        Parameters
        ----------
        i : Union[int, None]
            Iteration number or None.
            If initialized with `autosave_interval` as int value - specify current iteration number.
            If initialized with `autosave_interval` as timedelta value - specify None.
            If you specified `force=True` - this value is ignored.
        force : bool
            Specify True to force save state to disk.
            Default: False.
        """
        # Check if autosave is enabled
        if not self._autosave_enabled():
            return

        # Check if we should save now or not
        if (not force) and (not self.time_to_save(i)):
            return
        
        # Save internal state
        try:
            state_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'state.pt')
            self.save_state(state_file_name)
            if self.log:
                self.log.info(f'Saved state to {state_file_name}')
        except NotImplementedError:
            pass

        # Save model
        try:
            model_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'model.pt')
            self.save_model(model_file_name)
            if self.log:
                self.log.info(f'Saved model to {model_file_name}')
        except NotImplementedError:
            pass

        # Save chart
        try:
            chart_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'chart.png')
            self.save_chart(chart_file_name)
            if self.log:
                self.log.info(f'Saved chart to {chart_file_name}')
        except NotImplementedError:
            pass

        # Save config
        try:
            if not self._saved_config:
                config_file_name = os.path.join(self.autosave_dir, self.autosave_prefix + 'conf.txt')
                config = self.config
                if isinstance(config, str) and (len(config) > 0):
                    self.save_config(config_file_name)
                    if self.log:
                        self.log.info(f'Saved config to {config_file_name}')
        except NotImplementedError:
            pass

        # Update time info
        self.last_save_time = datetime.now()
    
    def load_state_dict(self, state, strict=False):
        attr_names = set(self.state_dict().keys())
        if strict:
            assert all(k in attr_names for k in state), \
                AssertionError(str(attr_names) + '\n' + str(state.keys()))
        for parameter, value in state.items():
            if parameter in attr_names:
                setattr(self, parameter, value)
    
    def load_state(self, filename, strict=False):
        self.load_state_dict(torch.load(filename), strict=strict)
    
    def save_state(self, filename):
        try:
            os.replace(filename, filename + '.bak')
        except FileNotFoundError:
            pass
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, strict=True):
        self.load_model_state(torch.load(filename), strict=strict)

    def save_model(self, filename):
        try:
            os.replace(filename, filename + '.bak')
        except FileNotFoundError:
            pass
        torch.save(self.model_state(), filename)
    
    def save_chart(self, filename):
        fig = None
        try:
            fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.grid(True, axis='both')
            self.draw_chart(ax1)
            fig.tight_layout()
            fig.savefig(filename)
        except NotImplementedError:
            pass
        except Exception as ex:
            self.log.error(f'Failed to save chart to {filename}: {str(ex)}')
        finally:
            if fig is not None:
                plt.close(fig)

    def save_config(self, filename):
        try:
            os.replace(filename, filename + '.bak')
        except FileNotFoundError:
            pass
        config = self.config
        if isinstance(config, str) and (len(config) > 0):
            with open(filename, 'w') as file:
                file.write(config)
                self._saved_config = True

    def get_params(self, deep=False):
        return dict(
            autosave_dir=self.autosave_dir,
            autosave_prefix=self.autosave_prefix,
            autosave_interval=self.autosave_interval,
        ) if deep else {}
    
    def extra_repr(self):
        params = self.get_params()
        return ',\n'.join(['  {}={}'.format(k, v.__name__ if hasattr(v, '__name__') else repr(v))
                          for k, v in params.items()])
    
    def __repr__(self):
        return f'{self.__class__.__name__}(\n{self.extra_repr()}\n)'
    
    @property
    def config(self) -> Optional[str]:
        return None

    def state_dict(self) -> Mapping:
        raise NotImplementedError()

    def model_state(self) -> Any:
        raise NotImplementedError()

    def load_model_state(self, model_state: Any, strict=False):
        raise NotImplementedError()

    def draw_chart(self, ax: matplotlib.pyplot.Axes):
        raise NotImplementedError()
