from __future__ import annotations
from typing import Union, Sequence, Callable, MutableMapping
from datetime import timedelta
from numbers import Real
import logging
import numpy as np
import torch
import math

from optim.autosaver import Autosaver


class DCEM(Autosaver):
    """
    The Differentiable Cross-Entropy Method (DCEM)
    
    Notes
    -----
    https://arxiv.org/pdf/1909.12830.pdf
    """
    
    def __init__(self,
                 initial_point: MutableMapping[str, torch.Tensor],
                 scoring_function: Callable[[Sequence[MutableMapping[str, torch.Tensor]], DCEM], Sequence[Real]],
                 population_size: Union[int, str] = 'auto',
                 selection_size: Union[int, str] = 'auto',
                 autosave_dir: Union[str, None] = '.',
                 autosave_prefix: Union[str, None] = None,
                 autosave_interval: Union[int, timedelta, None] = 5,
                 log: Union[logging.Logger, str, None] = None):
        super().__init__(autosave_dir, autosave_prefix, autosave_interval, log)
        # Process and save arguments
        assert callable(scoring_function)
        self.scoring_function = scoring_function
        
        # Compute total number of point's dimensions
        N = sum([x.numel() for x in initial_point.values()])
        
        if population_size == 'auto':
            population_size = 4 + int(3 * math.log(N))
        assert isinstance(population_size, int) and (population_size > 1)
        self.population_size = population_size

        if selection_size == 'auto':
            selection_size = int(population_size // 2)
        assert isinstance(selection_size, int) and (selection_size > 1) and (selection_size <= population_size)
        self.selection_size = selection_size
        
        # Get dtype and device from the first value
        x = next(iter(initial_point.values()))
        dtype = x.dtype
        device = x.device
        
        # Weights for best codes
        self.weights = torch.arange(1, selection_size + 1, dtype=dtype, device=device)
        self.weights = math.log(selection_size + 1 / 2) - torch.log(self.weights)
        self.weights = self.weights / self.weights.sum()
        
        # Variance-effective selection size
        mueff = self.weights.sum() ** 2 / (self.weights**2).sum()
        self.mueff = mueff
        
        # Time constant for cumulation for C
        self.cc = (4 + mueff/N) / (N+4 + 2*mueff/N)
        
        # Time constant for cumulation for sigma control
        self.cs = (mueff + 2)/(N + mueff + 5)
        
        # Learning rate for rank-one update of C
        self.c1 = 2 / ((N + 1.3)**2 + mueff)
        
        # Learning rate for rank-mu update of C
        self.cmu = 2*(mueff - 2 + 1/mueff) / ((N + 2)**2 + 2*mueff/2)
        
        # Damping for sigma
        self.damps = 1 + 2 * max(0, math.sqrt((mueff - 1)/(N + 1)) - 1) + self.cs
        
        # Total iterations counter
        self.n_iter = 0
        
        self.mean = {}
        self.pc = {}
        self.ps = {}
        self.B = {}
        self.D = {}
        self.C = {}
        self.c_iter = {}
        self.chiN = {}
        self.sigma = {}
        
        for name, x in initial_point.items():
            # Get number of elements in a variable x
            n = x.numel()

            # Step size (~ learning rate)
            self.sigma[name] = 0.5

            # Initial mean value
            self.mean[name] = x.clone()
            
            # Evolution path for C
            self.pc[name] = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
    
            # Evolution path for sigma
            self.ps[name] = torch.zeros((n, 1), dtype=x.dtype, device=x.device)
            
            # B defines the coordinate system
            self.B[name] = torch.eye(n, dtype=x.dtype, device=x.device)
            
            # D defines the scaling
            self.D[name] = torch.eye(n, dtype=x.dtype, device=x.device)
            
            # Covariance matrix
            self.C[name] = torch.eye(n, dtype=x.dtype, device=x.device)
            
            # Last iteration when B, D and C were updated
            self.c_iter[name] = 0
            
            # % expectation of || N(0, I) || == norm(randn(N, 1))
            self.chiN[name] = (n**0.5) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        self.stat = {}
    
    def fit(self, max_iter=100):
        self.autoload()
        while self.n_iter < max_iter:
            self.iterate()
            if self.time_to_save(self.n_iter):
                self.autosave()
            # TODO: Add condition to exit
            
    def iterate(self):
        # Construct codes of new generation
        points = [{} for _ in range(self.population_size)]
        drifts = [{} for _ in range(self.population_size)]
        for name, mean in self.mean.items():
            n = mean.numel()
            BD = torch.matmul(self.B[name], self.D[name])
            Z = torch.randn((n, self.population_size), dtype=mean.dtype, device=mean.device)
            D = torch.matmul(BD, Z)
            for i in range(self.population_size):
                # Initialize new point as a random drift from the current mean value
                point, drift = points[i], drifts[i]
                d = D[:, i]
                x = mean.clone() + self.sigma[name] * d.view_as(mean)
                point[name] = x
                drift[name] = Z[:, i]     # TODO: check if here must be d or z!
                
        # Evaluate new generation
        scores = self.scoring_function(points, self)
        assert isinstance(scores, Sequence) and (len(scores) == len(points))
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
            self.log.info(f'Iter: {self.n_iter}, perf: {values[2]}')

        # Perform selection
        selection = [(X, Z) for _, X, Z in sorted(zip(scores, points, drifts), key=lambda p: p[0], reverse=True)]
        selection = selection[:self.selection_size]
        points, drifts = zip(*selection)
        
        # Increase iterations counter
        self.n_iter += 1

        # Compute new weighted mean from selection
        for name, mean in self.mean.items():
            # Get parameters for `name`
            n = mean.numel()
            ps = self.ps[name]
            pc = self.pc[name]
            B = self.B[name]
            D = self.D[name]
            C = self.C[name]
            sigma = self.sigma[name]
            chiN = self.chiN[name]

            # Compute new mean value of x
            meanx = torch.zeros_like(mean)
            for i, point in enumerate(points):
                meanx += self.weights[i] * point[name]      # FIXME  Change to matrix form of X
            self.mean[name] = meanx
            
            # Compute mean value of gradients
            meanz = torch.zeros((n, 1), dtype=mean.dtype, device=mean.device)
            gradz = torch.zeros((n, len(drifts)), dtype=mean.dtype, device=mean.device)
            for i, drift in enumerate(drifts):
                z = drift[name]
                gradz[:, i] = z.view(-1)             # FIXME  Change to matrix form of Z
                meanz += self.weights[i] * z.view_as(meanz)
                
            # Cumulation: Update evolution paths
            ps = (1 - self.cs)*ps + math.sqrt(self.cs*(2-self.cs)*self.mueff) * torch.matmul(B, meanz)
            hs = torch.norm(ps) / math.sqrt(1 - (1 - self.cs)**(self.n_iter + 1)) / chiN < (1.4 + 2/(n + 1))
            pc = (1 - self.cc)*pc + hs*math.sqrt(self.cc*(2-self.cc)*self.mueff) * torch.matmul(torch.matmul(B, D), meanz)
            self.ps[name] = ps
            self.pc[name] = pc
            
            # Adapt covariance matrix C
            BDZ = torch.matmul(torch.matmul(B, D), gradz)
            C = (
                (1 - self.c1 - self.cmu) * C                # take values from old matrix
                + self.c1 * (
                    torch.matmul(pc, pc.t())                # plus rank-1 update
                    + (1 + hs) * self.cc*(2-self.cc) * C    # with minor correction
                )
                + self.cmu * torch.matmul(torch.matmul(BDZ, torch.diag(self.weights)), BDZ.t())
            )
            self.C[name] = C

            # Adapt step-size sigma
            sigma = sigma * math.exp((self.cs / self.damps) * (torch.norm(ps) / chiN - 1))
            self.sigma[name] = sigma
            
            # Update B and D from C
            if self.n_iter - self.c_iter[name] > self.population_size/(self.c1 + self.cmu)/n/10:
                C = torch.triu(C) + torch.triu(C, diagonal=1)   # Enforce symmetry
                D, B = torch.eig(C, eigenvectors=True)          # Get eigenvalues D and eigenvectors B
                D = torch.diag(torch.sqrt(D[0]))                # Use only real and ignore imaginary parts
                self.B[name] = B
                self.D[name] = D
                self.C[name] = C
                self.c_iter[name] = self.n_iter
                
    def draw_chart(self, ax1):
        if len(self.stat[0]) < 2:
            return
        x = list(range(len(self.stat[0])))
        ax1.fill_between(x, self.stat[0], self.stat[100], alpha=0.1)
        ax1.fill_between(x, self.stat[25], self.stat[75], alpha=0.5)
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
            'mean': self.mean,
            'pc': self.pc,
            'ps': self.ps,
            'B': self.B,
            'D': self.D,
            'C': self.C,
            'chiN': self.chiN,
            'sigma': self.sigma,
            'stat': self.stat,
        }
