"""
    Implementation of flexible standardize PyTorch module
    used for online standardization/normalization of input values
    via running mean and variance computation.
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: Dec 2020 - July 2021
    License: MIT
"""
from typing import Optional
import torch as th
import torch.nn as nn
import multiprocessing as mp


class Standardize(nn.Module):
    """
    Performs standardization/normalization of input tensors via running mean and variance
    
    Notes
    -----
    
    During training a running mean and variance (std) are calculated.
    Output is calculated as:
        if center=True:
            output = (input - mean)/(std + eps)
        otherwise:
            output = input/(std + eps)
        , where mean and std are calculated only for C dimension (features/channels).

    This class can work either in batch or single sample modes.
    It expects input to be one of any shape, for example:
    - (C,) - single vector sample with dimension C equal to `num_features`.
    - (N,C) - N-batch of vector samples.
    - (N,C,L) - N-batch of samples, each having C vectors of length L.
                For example: C spectrums of different sound frequencies of size L.
    - (N,C,H,W) - N-batch of images with C-colour channels, height H and width W.
    - (S,N,C) - S-sequence of N-batches of vector samples of size C.
    - (S,N,C,H,W) - S-sequence of N-batches of images with C-colour channels, height H and width W.
    - etc.

    If there are more dimensions after C - they are averaged in C dimension.
    For example: an RGB image (3,H,W) will be averages on its colour planes to get 3 values.
    
    If there are some dimensions before C - they will define the number of samples given,
    and they will also be averaged.
    
    Mean and std are calculated via `running mean and variance algorithm` by Knuth.
    This is a Welford's online algorithm for computing mean and variance,
    with some tricks to improve numerical stability.
    
    Parameters
    ----------
    num_features : int
        Number of features in specified dimension `dim`.
        Examples:
        1. If your input is (N, C), where N is batch, and dim=-1 - then num_features = C.
           In this case each element in C is being normalized across all element values
           passed through Standardize module.
        2. If your input is (N, C, L), where N is batch, and dim=-2 - then num_features = L.
           In this case each element in C is being normalized across mean values of L,
           passed through Standardize module.
        3. If your input is image (N, C, H, W), where N is batch, and dim=(-2, -1) - then num_features = H*W.
        4. If your input is image (N, H, W, C), where N is batch, and dim=-1 - then num_features = C.
    momentum : float
        Smoothing coefficient for exponential averaging.
        Default: 0.0001
    center : bool
        Whether to subtract mean value or not.
        If True: output = (input - mean) / (std + eps)
        If False: output = input / (std + eps)
        Default: True
    dim : int
        Specify dimension to standardize.
        Typically, you may want to specify negative index, in order to use Standardize module
        for both batch inputs and sequential batch inputs.

        Example 1: your input is an image (C, H, W) so you set dim=-3.
            In this case module takes average per-channel value across dimensions -1 and -2: H*W.
            Also you can feed in both batch inputs: (N, C, H, W), and sequential batch inputs: (S, N, C, H, W)
        Default: -1
    mask: Optional[torch.BoolTensor]
        Optionally specify a boolean vector of length `num_features`.
        Elements with False value will NOT be standardized.
        Default: None
    multiprocess_lock : Optional[multiprocessing.synchronize.Lock]
        Specify a Lock object if you need Standardize module to work in multiprocess environment.
        Default: None
    """

    eps = 1e-08
    mps = 1e+10
    
    def __init__(self,
                 num_features: int,
                 momentum=0.0001,
                 center=True,
                 dim: int = -1,
                 mask: Optional[th.BoolTensor] = None,
                 # multiprocess_lock: Optional[Lock] = None,
                 synchronized=False,
                 **kwargs):
        super().__init__()
        self.num_features = num_features
        self.period = int(2 / momentum)
        self.momentum = momentum
        self.center = center
        self.dim = dim
        assert ((mask is None) or (th.is_tensor(mask) and
                                   (mask.dtype == th.bool) and
                                   (mask.dim() == 1) and
                                   (mask.numel() == num_features))
                ), 'mask must be None or a tensor of dtype=bool'
        self.mask: Optional[th.BoolTensor] = mask
        self.synchronized = synchronized
        self.lock: Optional[mp.Lock] = mp.Lock() if synchronized else None
        self.register_buffer('n', tensor=None, persistent=True)
        self.register_buffer('mean', tensor=None, persistent=True)
        self.register_buffer('var', tensor=None, persistent=True)
        self.n: th.Tensor = th.tensor(0, dtype=th.int64)
        self.mean: th.Tensor = th.zeros(num_features)
        self.var: th.Tensor = th.zeros(num_features)
        if self.lock is not None:
            self.share_memory()

    def forward(self, input: th.Tensor) -> th.Tensor:
        # Get actual dimension index `dim` where samples are located
        ndims = input.dim()
        dim = (ndims + self.dim) if (self.dim < 0) else self.dim
        
        if self.lock is not None:
            self.lock.acquire()

        try:
            # Update running mean and var in training mode
            if self.training:
                all_dims = tuple(range(ndims))
                pre_dims, post_dims = all_dims[:dim], all_dims[dim + 1:]
                
                # Get samples truncated at dimension dim
                if len(post_dims) > 0:
                    s: th.Tensor = input.detach().mean(dim=post_dims)
                else:
                    s: th.Tensor = input.detach()

                # Get the number of samples
                k = max(1, input.shape[:dim].numel())

                # Calculate mean of all samples
                if len(pre_dims) > 0:
                    m: th.Tensor = s.mean(dim=pre_dims)
                else:
                    m = s
            
                # Update running mean
                assert (self.n == 0) or (m.shape == self.mean.shape)
                if (self.n == 0) or (self.mean.numel() == 0):
                    self.mean.copy_(m)
                    old_mean = m
                elif self.n < self.period:
                    #self.mean.mul_(self.n).add_(k * x).div_(self.n + k)
                    old_mean: th.Tensor = self.mean.clone()
                    self.mean.add_(k * (m - old_mean).div_(self.n + k))
                else:
                    # TODO: Take into account number of samples k
                    old_mean: th.Tensor = self.mean.clone()
                    self.mean.mul_(1 - self.momentum).add_(self.momentum, m)
                
                # Update running mean of variance
                # Get mean variance of k input samples
                if len(pre_dims) > 0:
                    expand_dims = (1,) * dim + (-1,)
                    var = (s - old_mean.view(expand_dims)) * (s - self.mean.view(expand_dims))
                    var: th.Tensor = var.mean(dim=pre_dims)
                else:
                    assert s.numel() == self.num_features
                    var: th.Tensor = (s - old_mean) * (s - self.mean)
                if (self.n == 0) or (self.var.numel() == 0):
                    self.var.copy_(var)
                elif self.n < self.period:
                    self.var.mul_(self.n).add_(k * var).div_(self.n + k)
                else:
                    # TODO: Take into account number of samples k
                    self.var.mul_(1 - self.momentum).add_(self.momentum, var)
    
                # Increment step counter, avoiding integer overflow
                # if self.n < th.iinfo(self.n.dtype).max - 1:
                self.n += k
            
            # Get mean and std
            mean: th.Tensor = self.mean
            std: th.Tensor = self.var.sqrt()
            
        finally:
            if self.lock is not None:
                self.lock.release()

        # Compute bias correction due to initial zero mean and variance
        if self.n < self.period:
            b = 1.0 / (1.0 - (1.0 - self.momentum) ** max(1, self.n.item()))
        else:
            b = 1.0

        # Fix NaN or small values - for numerical stability
        std[std != std] = b
        std[std < self.eps] = b
        assert (self.n == 0) or th.isfinite(std).all()
        
        # Apply mask if any
        if self.mask is not None:
            mean = mean * self.mask
            std[~self.mask] = 1.0

        # Calculate output = (input - mean) / (std + eps)   or   output = input / (std + eps)
        expand_dims = (1,)*dim + (self.num_features,) + (1,)*(ndims - dim - 1)
        mean = mean.view(expand_dims)
        std = std.view(expand_dims)
        delta = (input - mean) if self.center else input
        output = delta.div(std + self.eps)
        
        return output
    
    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'num_features={self.num_features}, '
            f'momentum={self.momentum}, '
            f'center={self.center})'
        )

    
if __name__ == '__main__':
    print('Basic test:')
    inputs = th.tensor([
        [0, -1, 1],
        [1,  1, 1],
        [0, -1, 1],
        [1,  1, 1],
        [0, -1, 1],
        [1,  1, 1],
        [0, -1, 1],
        [1, 1, 1],
    ], dtype=th.float32)
    s = Standardize(num_features=3, momentum=0.0001)
    for i in range(len(inputs)):
        x = inputs[i]
        y = s.forward(x.unsqueeze(0)).squeeze()
        print(x, y)

    print('--------')
    s = Standardize(num_features=3, momentum=0.0001, center=False)
    for i in range(len(inputs)):
        x = inputs[i]
        y = s.forward(x.unsqueeze(0)).squeeze()
        print(x, y)

    print('--------')
    s = Standardize(num_features=3, momentum=0.0001, center=False)
    inputs = inputs * 4
    for i in range(len(inputs)):
        x = inputs[i]
        y = s.forward(x.unsqueeze(0)).squeeze()
        print(x, y)
