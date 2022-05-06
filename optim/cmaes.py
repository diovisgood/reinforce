"""
    Implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES) in Pytorch
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - June 2021
    License: MIT
"""
from __future__ import annotations
from typing import Union, Sequence, Callable, Optional, Tuple, Any
from datetime import timedelta
from numbers import Real
import logging
import numpy as np
import torch as th
import math

from optim.autosaver import Autosaver


class CMAES(Autosaver):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    
    Notes
    -----
    This class is based on the paper:
    [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/pdf/1604.00772.pdf)
    
    Parameters
    ----------
    initial_point : torch.Tensor
        Initial point is a vector of parameters in a high dimensional space.
        It is also called initial `solution`.
        The goal of evolution algorithm is to find another, `optimal solution`, starting from the initial.
        A model with `optimal parameters` will have the highest performance.
    scoring_function : Callable[[Sequence[torch.Tensor], CMAES], Sequence[Real]]
        Specify a function to evaluate a set of different `solutions` - the current population.
        This is the most important function for any evolution algorithm, including CMA-ES.
        It takes as input a list of different `solutions`, evaluates each `solution`
        and computes its score - a floating point value.
        The larger the score - the better the solution is.
        In the end it must return a list of scores for all `solutions`.
    population_size : Union[int, str]
        Specify population size or `auto` - to let algorithm choose optimal size.
        Population size defines how many different solutions to try on each iteration.
        Default: 'auto'
    selection_size : Union[int, str]
        Specify selection size or `auto` - to let algorithm choose optimal size.
        Selection size defines how many best solutions to select after each iteration
        to compute the next best solution.
        This value must be less than `population_size`.
        Default: 'auto'
    initial_sigma : float
        Initial step size (~ learning rate).
        Learning rate in CMAES algorithm is computed and updated automatically on each iteration.
        But this value is used as initial one.
        Default: 1.5
    tolx : Union[float, None, str]
        Specify tolerance stopping criterion or `auto` - to let algorithm choose optimal value.
        Training is stopped if the std of the normal distribution of weights becomes smaller than `tolx`.
        Default: 'auto'
    tolxupper : float
        Specify upper tolerance stopping criterion or `auto` - to let algorithm choose optimal value.
        This value is used to detect divergent behavior.
        Default: 1e4
    tolconditioncov : float
        Stop if the condition number of the covariance matrix exceeds `tolconditioncov`
        Default: 1e14
    repair_parameters : bool
        Sometimes during training some weights can become zero or infinitely large or NaN.
        Turn on this option to automatically detect and fix these weights.
        When True, after every iteration will perform `reparation` of broken parameters.
        In this case they are replaced with some small random values.
        Default: True
    bounds: Optional[Tuple[Real, Real]]
        You may want to specify (min, max) values to limit your parameters to some diapason.
        Default: None
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
        some debug, information or warnings from CMAES instance.
        Default: None
    """
    
    _eps = 1e-8

    def __init__(self,
                 initial_point: th.Tensor,
                 scoring_function: Callable[[Sequence[th.Tensor], CMAES], Sequence[Real]],
                 population_size: Union[int, str] = 'auto',
                 selection_size: Union[int, str] = 'auto',
                 initial_sigma: float = 1.5,
                 tolx: Union[float, None, str] = 'auto',
                 tolxupper: float = 1e4,
                 tolconditioncov: float = 1e14,
                 repair_parameters=True,
                 bounds: Optional[Tuple[Real, Real]] = None,
                 autosave_dir: Optional[str] = '.',
                 autosave_prefix: Optional[str] = None,
                 autosave_interval: Optional[Union[int, timedelta]] = 5,
                 log: Union[logging.Logger, str, None] = None):
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        # Process and save arguments
        assert callable(scoring_function)
        self.scoring_function = scoring_function
    
        # Compute total number of point's dimensions
        x = initial_point
        n = x.numel()
    
        if population_size == 'auto':
            population_size = 4 + int(3 * math.log(n))
        assert isinstance(population_size, int) and (population_size > 1)
        self.population_size: int = population_size
    
        if selection_size == 'auto':
            selection_size = int(population_size // 2)
        assert isinstance(selection_size, int) and (selection_size > 1) and (selection_size <= population_size)
        self.selection_size: int = selection_size
        
        if tolx == 'auto':
            tolx = 1e-12 * initial_sigma
        else:
            assert (tolx is None) or (isinstance(tolx, float) and (tolx < initial_sigma))
        self.tolx: float = tolx
        
        assert isinstance(tolxupper, Real) and (tolxupper > initial_sigma)
        self.tolxupper: float = tolxupper

        assert isinstance(tolconditioncov, float) and (tolconditioncov > initial_sigma)
        self.tolconditioncov: float = tolconditioncov
        
        self.repair_parameters = repair_parameters
        
        assert (bounds is None) or (isinstance(bounds, Tuple) and (len(bounds) >= 2))
        self.bounds: Optional[Tuple[float, float]] = bounds

        # Get dtype and device from the first floating-point value
        if th.is_floating_point(x):
            dtype, device = x.dtype, x.device
        else:
            dtype, device = th.get_default_dtype(), 'cpu'
            
        # Weights for best codes,  (selection_size, 1)
        self.weights = th.arange(1, selection_size + 1, dtype=dtype, device=device)
        self.weights = math.log(selection_size + 1 / 2) - th.log(self.weights)
        self.weights = self.weights / self.weights.sum()
        self.weights = self.weights.unsqueeze(dim=0)
    
        # Variance-effective selection size
        mueff = self.weights.sum() ** 2 / (self.weights ** 2).sum()
        self.mueff = mueff
    
        # Total iterations counter
        self.n_iter = 0
    
        alpha_cov = 2
    
        # Initial mean value
        self.mean = x.clone()
    
        # Step size (~ learning rate)
        sigma = initial_sigma
        self.sigma = sigma
    
        # Learning rate for the cumulation for the step-size control (eq.55)
        cs = (mueff + 2) / (n + mueff + 5)
        self.cs = cs
        assert (cs < 1), 'Invalid learning rate for cumulation for the step-size control'
    
        # Damping for sigma
        ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        self.ds = ds
    
        # Learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        self.cc = cc
    
        # Learning rate for rank-one update of C
        c1 = alpha_cov / ((n + 1.3) ** 2 + mueff)
        self.c1 = c1
    
        # Learning rate for rank-mu update of C
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large population_size
            alpha_cov * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + alpha_cov * mueff / 2)
        )
        self.cmu = cmu
    
        assert c1 <= 1 - cmu, 'Invalid learning rate for the rank-one update'
        assert cmu <= 1 - c1, 'Invalid learning rate for the rank-μ update'
        
        # Evolution path for C
        self.pc = th.zeros((1, n), dtype=x.dtype, device=x.device)
    
        # Evolution path for sigma
        self.ps = th.zeros((1, n), dtype=x.dtype, device=x.device)
    
        # B defines the coordinate system (eigenvectors), (n, n)
        self.B = th.eye(n, dtype=x.dtype, device=x.device)
    
        # D defines the scaling (eigenvalues) vector of size: (n)
        self.D = th.ones(n, dtype=x.dtype, device=x.device)
    
        # Covariance matrix (basically, its upper part as a vector of size: n * (n + 1) // 2)
        self.C = triu_to_vector(th.eye(n, dtype=x.dtype, device=x.device))
    
        # Last iteration when B, D and C were updated
        self.c_iter = 0
    
        # Expectation of || N(0, I) || == norm(randn(N, 1))
        self.chiN = (n ** 0.5) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.stat = {}
        
        # Try to load previously saved state
        self.autoload()

    def fit(self, max_iter=100):
        while self.n_iter < max_iter:
            if self.iterate():
                break
            self.autosave(self.n_iter)
        self.autosave(self.n_iter, force=True)
            
    def iterate(self) -> bool:
        # Increase iterations counter
        self.n_iter += 1

        # Construct new generation
        if self.log:
            self.log.info(f'Iteration {self.n_iter}. Generating {self.population_size} new entities...')
        population = []
        mean_old = self.mean
        n = mean_old.numel()
        B = self.B
        D = self.D
        BD = B @ th.diag(D)   # BD = B x D, (n, n)
        assert th.isfinite(BD).all()
        Z = th.randn((self.population_size, n), dtype=mean_old.dtype, device=mean_old.device)  # (pop_size, n)
        Y = Z @ BD  # (pop_size, n)
        del BD, Z
        assert th.isfinite(Y).all()
        sigma = self.sigma
        X = mean_old.view(1, -1) + sigma * Y  # (pop_size, n)
        assert th.isfinite(X).all()
        for i in range(self.population_size):
            population.append(X[i, :].view_as(mean_old))
            
        # Evaluate new generation
        if self.log:
            self.log.info(f'Iteration {self.n_iter}. Evaluating new entities...')
        scores = self.scoring_function(population, self)
        assert isinstance(scores, Sequence) and (len(scores) == len(population))
        assert all([isinstance(score, Real) for score in scores])
        
        # Write down some statistics
        percentiles = [0, 25, 50, 75, 100]
        values = np.percentile(scores, q=percentiles)
        for percent, score in zip(percentiles, values):
            if percent not in self.stat:
                self.stat[percent] = []
            self.stat[percent].append(score)

        # Print epoch results
        if self.log:
            self.log.info(f'Iteration {self.n_iter}. Median performance: {values[2]}')

        # Perform selection
        selection_index = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        selection_index = selection_index[:self.selection_size]
        Y = Y[selection_index, :]
        X = X[selection_index, :]
        
        if self.log:
            self.log.info(f'Iteration {self.n_iter}. Updating parameters...')

        # Get parameters
        mueff = self.mueff
        pop_size = self.population_size
        sel_size = self.selection_size
        cc = self.cc
        cs = self.cs
        ds = self.ds
        c1 = self.c1
        cmu = self.cmu
        ps = self.ps
        pc = self.pc
        C = triu_to_symmetric(vector_to_triu(self.C, n))
        chiN = self.chiN

        # Compute new weighted mean from selection

        # Compute new mean value of x
        mean_new = self.weights @ X      # (1, n)
        self.mean = mean_new.view_as(mean_old)
        if self.repair_parameters:
            self.perform_repair_parameters()
        assert th.isfinite(mean_new).all()
            
        # Compute mean value of drift y
        mean_y = self.weights @ Y  # (1, n)  eq.41
        assert th.isfinite(mean_y).all()

        # Cumulation: Update evolution paths
        C_2 = B @ th.diag(1 / D) @ B.T   # C^(-1/2) = B D^(-1) B^T    (n, n)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (mean_y @ C_2)  # (1, n)
        assert th.isfinite(ps).all()
        self.ps = ps
        ps_norm = th.norm(ps)
        assert th.isfinite(ps_norm).all()
        del C_2
            
        # Adapt step-size sigma
        sigma = sigma * math.exp((cs / ds) * (ps_norm / chiN - 1))
        assert th.isfinite(th.tensor(sigma)).all()
        self.sigma = sigma
            
        # (eq.45)
        hs_cond_left = ps_norm / math.sqrt(1 - (1 - cs)**(2 * self.n_iter / pop_size))  # TODO: Test pop_size
        hs_cond_right = (1.4 + 2 / (n + 1)) * chiN
        hs = 1.0 if (hs_cond_left < hs_cond_right) else 0.0
        del hs_cond_left, hs_cond_right

        # (eq.45)
        if hs == 1:
            pc = (1 - cc) * pc + math.sqrt(cc * (2 - cc) * mueff) * mean_y  # (1, n)
        else:
            pc = (1 - cc) * pc
        assert th.isfinite(pc).all()
        self.pc = pc
            
        # Adapt covariance matrix C
        rank_one = pc.T @ pc    # (n, n)
        assert th.isfinite(rank_one).all()
        rank_mu = mean_y.T @ mean_y  # (n, n)
        assert th.isfinite(rank_mu).all()

        delta_hs = (1 - hs) * cc * (2 - cc)  # (p.28)
        assert delta_hs <= 1
        
        # Update covariance matrix
        C *= (1 + c1 * delta_hs - c1 - cmu)
        C += c1 * rank_one
        C += cmu * rank_mu
        self.C = triu_to_vector(C)
        
        # Free unused variables and memory
        del delta_hs, rank_one, rank_mu, mean_y
        
        # Update eigenvectors B and eigenvalues D from covariance matrix C
        if self.n_iter - self.c_iter > self.population_size/(c1 + cmu)/n/10:
            C = (C + C.T) / 2    # Enforce symmetry
            D, B = th.eig(C, eigenvectors=True)  # Get eigenvalues D and eigenvectors B
            # D = th.sqrt(D[:, 0]).squeeze()  # Use only real and ignore imaginary parts of eigenvalues
            D = D[:, 0].squeeze()  # Use only real and ignore imaginary parts of eigenvalues
            D = th.sqrt(th.clamp(D, min=self._eps))
            assert B.shape == (n, n)
            assert D.shape == (n,)
            assert th.isfinite(B).all()
            assert th.isfinite(D).all()
            self.B = B  # (n, n)
            self.D = D  # (n,)
            self.C = triu_to_vector(C)
            self.c_iter = self.n_iter
        
        # Stop if the std of the normal distribution is smaller than tolx
        # in all coordinates and pc is smaller than tolx in all components.
        diagC = th.diag(C)
        del C
        if (sigma * diagC < self.tolx).all() and (sigma * pc < self.tolx).all():
            return True

        # Stop if detecting divergent behavior.
        if sigma * D.max() > self.tolxupper:
            if self.log:
                self.log.warning('Either initial σ is too small or divergence detected!')
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if (mean_new == mean_new + (0.2 * sigma * th.sqrt(diagC))).any():
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.n_iter % n
        if (mean_new == mean_new + (0.1 * sigma * D[i] * B[:, i]).unsqueeze(0)).all():
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        if (D.max() / D.min()) > self.tolconditioncov:
            return True
        
        return False
        
    def perform_repair_parameters(self):
        self.mean[(self.mean != self.mean) + (self.mean == 0)] = np.random.randn()/self.mean.numel()
        if self.bounds is not None:
            self.mean = self.mean.clamp(min=self.bounds[0], max=self.bounds[1])

    def draw_chart(self, ax1):
        if len(self.stat[0]) < 2:
            return
        x = list(range(len(self.stat[0])))
        ax1.fill_between(x, self.stat[0], self.stat[100], color='red', alpha=0.1, linewidth=0, label='scores 0..100%')
        ax1.fill_between(x, self.stat[25], self.stat[75], color='red', alpha=0.2, linewidth=0, label='scores 25..75%')
        ax1.plot(x, self.stat[50], label='median', color='red')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('score', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, axis='both')
        ax1.legend()

    def get_params(self, deep=True):
        return {
            'population_size': self.population_size,
            'selection_size': self.selection_size,
            **super().get_params(deep),
        }

    def state_dict(self):
        return {
            'n_iter': self.n_iter,
            'c_iter': self.c_iter,
            'sigma': self.sigma,
            'mean': self.mean,
            'pc': self.pc,
            'ps': self.ps,
            'B': self.B,
            'D': self.D,
            'C': self.C,
            'stat': self.stat,
        }

    def model_state(self):
        return self.mean.clone()

    def load_model_state(self, model_state: Any, strict=False):
        self.mean = model_state.clone()


def vector_to_triu(v: th.Tensor, n: int) -> th.Tensor:
    if not isinstance(n, int):
        n = (-1 + np.sqrt(1 + 8*len(v)))/2
    m = th.zeros(n, n, dtype=v.dtype, device=v.device)
    m[np.triu_indices(n)] = v
    return m


def triu_to_vector(m: th.Tensor) -> th.Tensor:
    row_idx, col_idx = np.triu_indices(m.shape[0])
    row_idx = th.LongTensor(row_idx, device=m.device)
    col_idx = th.LongTensor(col_idx, device=m.device)
    v = m[row_idx, col_idx]
    return v


def triu_to_symmetric(m: th.Tensor) -> th.Tensor:
    return th.triu(m) + th.triu(m, diagonal=1).T


def cov_to_symmetric(m: th.Tensor) -> th.Tensor:
    return (m + m.T) / 2
