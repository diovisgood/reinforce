import torch
import numpy as np


def vector_to_triu(v: torch.Tensor, n: int) -> torch.Tensor:
    if not isinstance(n, int):
        n = (-1 + np.sqrt(1 + 8*len(v)))/2
    m = torch.zeros(n, n, dtype=v.dtype, device=v.device)
    m[np.triu_indices(n)] = v
    return m


def triu_to_vector(m: torch.Tensor) -> torch.Tensor:
    row_idx, col_idx = np.triu_indices(m.shape[0])
    row_idx = torch.LongTensor(row_idx, device=m.device)
    col_idx = torch.LongTensor(col_idx, device=m.device)
    v = m[row_idx, col_idx]
    return v


def triu_to_symmetric(m: torch.Tensor) -> torch.Tensor:
    return torch.triu(m) + torch.triu(m, diagonal=1).t()


def cov_to_symmetric(m: torch.Tensor) -> torch.Tensor:
    return (m + m.t()) / 2


if __name__ == '__main__':
    n = 4
    m = torch.randn(n, n)
    print('random:\n', m)
    m = triu_to_symmetric(m)
    print('symmetric:\n', m, m.numel())
    v = triu_to_vector(m)
    print('vector:\n', v, v.numel())
    r = triu_to_symmetric(vector_to_triu(v, n))
    print('restored symmetric:\n', r)
