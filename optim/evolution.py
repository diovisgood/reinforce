from typing import Union, Sequence, Callable, Mapping, MutableMapping, Tuple, Any
from collections import namedtuple
from datetime import timedelta
from numbers import Real
import numpy as np
import torch
import logging
import random
import math
import copy
import os


Entity = namedtuple('Entity', 'code params score')


class Evolution:
    """
    Evolution Strategy with σ-Self-Adaptation Algorithm
    
    Notes
    -----
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    https://arxiv.org/pdf/1604.00772.pdf
    
    Hans-Georg Beyer, Hans-Paul Schwefel, Evolution strategies - A comprehensive introduction, 2002
    https://www.researchgate.net/publication/220132816_Evolution_strategies_-_A_comprehensive_introduction
    
    Hans-Georg Beyer, Toward a Theory of Evolution Strategies: Self-Adaptation
    ftp://ftp.dca.fee.unicamp.br/pub/docs/vonzuben/ia707_1s06/textos/beyer.pdf
    """
    
    def __init__(self,
                 initial_code: MutableMapping,
                 population_size: Union[int, str] = 'auto',
                 parents_number: Union[int, str] = 'auto',
                 selection_size: Union[int, str] = 'auto',
                 covariance_ema_factor: Union[float, str] = 'auto',
                 evaluate_function: Callable[[Sequence[MutableMapping]], Sequence[Real]] = None,
                 cross_breed_function: Callable[[Sequence[MutableMapping]], MutableMapping] = None,
                 mutate_function: Callable[[Sequence[MutableMapping]], MutableMapping] = None,
                 # mutation_factor: (float, Callable[[], Real]) = 0.2,
                 selection_includes_parents=True,
                 # learning_rate: float = 0.03,
                 autosave_dir: Union[str, None] = './',
                 autosave_prefix: Union[str, None] = None,
                 autosave_interval: Union[int, timedelta, None] = 5,
                 log: Union[logging.Logger, str, None] = None
                 ):
        # Initialize covariance matrix for each parameter with `name` and value of `x`
        N = 0
        self.covariance = {}
        self.step_size_factor = {}
        self.covariance_path_factor = {}
        self.covariance_selection_factor = {}
        for name, x in initial_code.items():
            n = self._get_flat_size(x)
            path_factor = 2 / n**2
            selection_factor = min(selection_size / n**2, 1 - path_factor)
            self.covariance[name] = np.identity(n)
            self.step_size_factor[name] = path_factor
            self.covariance_path_factor[name] = path_factor
            self.covariance_selection_factor[name] = selection_factor
            N += n
        
        if population_size == 'auto':
            population_size = 4 + int(3 * math.log(N))
        assert isinstance(population_size, int) and (population_size > 1)
        self.population_size = population_size

        if selection_size == 'auto':
            selection_size = int(population_size // 2)
        assert isinstance(selection_size, int) and (selection_size > 1) and (selection_size <= population_size)
        self.selection_size = selection_size

        if parents_number == 'auto':
            parents_number = selection_size
        assert isinstance(parents_number, int) and (parents_number >= 1) and (parents_number <= selection_size)
        self.parents_number = parents_number

        assert callable(evaluate_function)
        self.evaluate_function = evaluate_function
        
        assert callable(cross_breed_function)
        self.cross_breed_function = cross_breed_function
    
        # assert callable(mutation_factor) or (isinstance(mutation_factor, float) and (0 < mutation_factor < 1))
        # self.mutation_factor = mutation_factor
        
        # assert isinstance(learning_rate, float) and (0 < learning_rate <= 1)
        # self.learning_rate = learning_rate
        
        self.selection_includes_parents = selection_includes_parents
        
        assert (autosave_dir is None) or isinstance(autosave_dir, str)
        self.autosave_dir = autosave_dir
        assert (autosave_prefix is None) or isinstance(autosave_prefix, str)
        self.autosave_prefix = autosave_prefix
        assert (autosave_interval is None) or isinstance(autosave_interval, (int, timedelta))
        self.autosave_interval = autosave_interval
        assert (autosave_interval is None) or (isinstance(autosave_dir, str) and os.path.isdir(autosave_dir))

        assert (log is None) or isinstance(log, (str, logging.Logger))
        if log is None:
            log = str(self.__class__.__name__)
        if isinstance(log, str):
            log = logging.getLogger(log)
        self.log = log
        
        self.population = [Entity(code=initial_code, params=0, score=-math.inf)]
        self.stat = {}
        self.n_iter = 0
    
    @staticmethod
    def _get_flat_size(x: Union[np.ndarray, torch.nn.Parameter, torch.Tensor]) -> int:
        if isinstance(x, torch.nn.Parameter):
            return x.data.view(-1).numel()
        elif torch.is_tensor(x):
            return x.view(-1).numel()
        elif isinstance(x, np.ndarray):
            return x.flatten().size()
        else:
            raise ValueError(f'Unexpected element: {str(x)}')

    def iterate(self):
        # Construct entities of new generation
        codes = []
        params = []
        for i in range(self.population_size):
            # Choose one or several parents
            selection_size = min(self.selection_size, len(self.population))
            cross_breed_size = min(self.parents_number, len(self.population))
            parents_indexes = random.sample(range(selection_size), cross_breed_size)
            parent_codes = [self.population[p].code for p in parents_indexes]
            parent_params = [self.population[p].params for p in parents_indexes]
            # Create new entity's code
            code = self.cross_breed_function(parent_codes)
            # Create new entity's strategy parameters
            params = self.cross_breed_function()
            # Mutate strategy parameters
            params = self.mutate(params)
            # Mutate new code
            code = self.mutate(code, params)
            codes.append(code)
            params.append(params)

        # Evaluate new generation
        scores = self.evaluate_function(codes)
        assert isinstance(scores, Sequence) and (len(scores) == len(codes))
        assert all([isinstance(score, Real) for score in scores])
        generation = [Entity(code, params, score) for (code, params, score) in zip(codes, params, scores)]
        
        # Write down some statistics
        percentiles = [0, 25, 50, 75, 100]
        values = np.percentile(scores, q=percentiles)
        for percent, score in zip(percentiles, values):
            if percent not in self.stat:
                self.stat[percent] = []
            self.stat[percent].append(score)
        
        # Perform selection
        if self.selection_includes_parents:
            generation.extend(self.population)
        generation = sorted(generation, key=lambda a: a.score, reverse=True)
        self.population = generation[:self.selection_size]

    @staticmethod
    def perform_weight_decay(param_groups, weight_decay):
        if (weight_decay is None) or (weight_decay <= 0) or (weight_decay >= 1):
            return
        for group in param_groups:
            for param in group['params']:
                param.data = param.data.add(-weight_decay, param.data)
                
    def save_chart(self, chart_file_name):
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
            fig = plt.figure(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_xlabel('epoch')
            color_loss = 'tab:blue'
            ax1.set_yscale('log')
            ax1.set_ylabel('loss', color=color_loss)
            ax1.plot(self.train_loss_history, label='train', color=color_loss)
            ax1.plot(self.test_loss_history, label='test', color='tab:green')
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax1.legend()
            if len(self.performance_history) > 0:
                color_perf = 'tab:red'
                ax2 = ax1.twinx()
                ax2.set_ylabel('performance', color=color_perf)
                ax2.plot(self.performance_history, color=color_perf)
                ax2.tick_params(axis='y', labelcolor=color_perf)
            fig.tight_layout()
            fig.savefig(chart_file_name)
            plt.close(fig)
        except Exception as ex:
            self.log.error(f'Failed to save chart to {chart_file_name}: {str(ex)}')
            pass

    def get_params(self, deep=True):
        return {
            'autosave_dir': self.autosave_dir,
            'autosave_prefix': self.autosave_prefix,
            'autosave_interval': self.autosave_interval,
        }

    def state_dict(self):
        return {
            'trainer': {
                'n_iter': self.n_iter,
                'n_epoch': self.n_epoch,
                'best_test_loss': self.best_test_loss,
                'best_parameters': self.best_parameters,
                'no_improvement_count': self.no_improvement_count,
                'train_loss_history': self.train_loss_history,
                'test_loss_history': self.test_loss_history,
                'performance_history': self.performance_history,
            }
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
            assert all(k in state for k in ('model', 'optimizer', 'trainer'))
            required_trainer_keys = Trainer.state_dict(self)['trainer'].keys()
            assert all(k in required_trainer_keys for k in state['trainer']), \
                AssertionError(str(required_trainer_keys) + '\n' + str(state['trainer'].keys()))
        if 'model' in state:
            self.model.load_state_dict(state['model'], strict=strict)
        if 'optimizer' in state:
            if (state['optimizer'] is not None) and (self.optimizer is not None):
                self.optimizer.load_state_dict(state['optimizer'])
        if 'lr_scheduler' in state:
            if (state['lr_scheduler'] is not None) and (self.lr_scheduler is not None):
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        if 'trainer' in state:
            for parameter, value in state['trainer'].items():
                if parameter in {
                    'n_iter', 'n_epoch', 'best_test_loss', 'best_parameters',
                    'no_improvement_count', 'train_loss_history',
                    'test_loss_history', 'performance_history'
                }:
                    setattr(self, parameter, value)

    def save_state(self, filename):
        torch.save(self.state_dict(), filename)

    def load_state(self, filename, strict=False):
        state = torch.load(filename)
        self.load_state_dict(state, strict=strict)

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename, strict=False):
        model_state_dict = torch.load(filename)
        self.model.load_state_dict(model_state_dict, strict=strict)

    def extra_repr(self):
        params = self.get_params()
        return ', '.join(['{}={}'.format(k, v.__name__ if hasattr(v, '__name__') else str(v))
            for k, v in params.items()])

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            self.extra_repr()
        )


def mutate(values: MutableMapping) -> MutableMapping:
    new_values = copy.deepcopy(values)
    for name, value in values.items():
        new_values[name] = mutate_value(value)


def mutate_value(value: Union[np.ndarray, torch.nn.Parameter, torch.Tensor]
                 ) -> Union[np.ndarray, torch.nn.Parameter, torch.Tensor]:
    if isinstance(value, torch.nn.Parameter):
        g = torch.randn_like(value)
        new_entity[name].data += (g * mutation_factor)
    elif torch.is_tensor(weight):
        if weight.dtype.is_floating_point:
            g = torch.randn_like(weight)
            grad[name] = g
            new_entity[name] += (g * mutation_factor)
    elif isinstance(weight, np.ndarray):
        g = np.random.randn(*weight.shape)
        grad[name] = g
        new_entity[name] = (weight + (g * mutation_factor)).astype(weight.dtype)
    else:
        raise ValueError(f'Invalid entity value {str(name)}: {str(weight)}')
