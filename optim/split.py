"""
    Utility functions to split dataset into train and test parts
    Copyright: Pavel B. Chernov, pavel.b.chernov@gmail.com
    Date: September 2019 - October 2021
    License: MIT
"""

from typing import Union, Tuple, Any, Optional, Sequence, Dict, List
import sys
from datetime import date, timedelta
import numpy as np

try:
    import torch as th
except ImportError:
    # Create a fake torch class to avoid functions typing errors
    class _fake_torch(object):
        pass
    th = _fake_torch()
    th.Tensor = None

try:
    import pandas as pd
except ImportError:
    # Create a fake pandas class to avoid functions typing errors
    class _fake_pandas(object):
        pass
    pd = _fake_pandas()
    pd.Series = None


def train_test_split(n: int,
                     test_fraction=0.1,
                     random_seed: Optional[Any] = None,
                     shuffle=True
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a number of samples `n` splits sequence [0...n] into train and test parts

    Parameters
    ----------
    n : int
        Number of samples in dataset
    test_fraction : float number (0..1)
        Specifies the ratio of test/train size
    random_seed : Optional[Any]
        Specify a seed value for `np.random.seed()`
        train and test indexes are randomly shuffled.
        You need to use the same seed value to get the same train/test indexes at different runs.
    shuffle : bool
        Specify True to shuffle train and test indices. Default: True

    Return
    ------
    (train_indexes, test_indexes): Tuple of two ndarrays
        train indexes and test indexes
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    index = np.random.permutation(n) if shuffle else np.arange(n)
    split = min(n, max(1, int(n * test_fraction)))
    return index[split:], index[:split]


def stratified_split(y: Union[th.Tensor, np.ndarray, pd.Series, Sequence],
                     test_fraction=0.1,
                     compute_class_weights=True,
                     random_seed: Optional[Any] = None,
                     shuffle=True
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[Any, float]]]:
    """
    Given only target labels y of a dataset splits it into train and test parts,
    so that both parts contain nearly equal proportion of target classes.

    Parameters
    ----------
    y : torch.Tensor or pandas.Series or numpy.ndarray or list
        Sequence which contains target classes.
        For multiclass classification this should be a matrix with rows as sample indexes
        and columns as different classes.
    test_fraction : float number (0..1)
        Specifies the ratio of test/train size
    compute_class_weights : bool
        Set True to get class weights as the third element in return tuple.
    random_seed : Optional[Any]
        Specify a seed value for `np.random.seed()`
        train and test indexes are randomly shuffled.
        You need to use the same seed value to get the same train/test indexes at different runs.
    shuffle : bool
        Specify True to shuffle train and test indices. Default: True

    Returns
    -------
    (train_indexes, test_indexes, [class_weights]) : Tuple[np.ndarray, np.ndarray, Optional[Dict[Any, float]]]
        train_indexes - First array is train indexes.
        test_indexes - Second array is test indexes.
        class_weights - Third (optional) is a dict with class weights.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if isinstance(y, np.ndarray):
        pass
    elif ('torch' in sys.modules) and th.is_tensor(y):
        y = y.numpy()
    elif ('pandas' in sys.modules) and isinstance(y, pd.Series):
        y = y.to_numpy()
    elif isinstance(y, Sequence):
        y = np.array(y)
    else:
        raise ValueError('y can be `numpy.ndarray`, `torch.Tensor`, `pandas.Series` or `list`')
    if y.ndim > 2:
        # No stratification for > 2 dimension
        train_indexes, test_indexes = train_test_split(
            len(y),
            test_fraction=test_fraction,
            random_seed=random_seed
        )
        return (train_indexes, test_indexes, None) if compute_class_weights else (train_indexes, test_indexes)
    elif y.ndim == 2:
        # If y is a matrix - convert it to an array of strings
        y = np.array([' '.join(row.astype('str')) for row in y])
    # Get unique values (=classes) in vector y and their indexes
    classes, indexes = np.unique(y, return_inverse=True)
    # Calculate count of each unique value (=class)
    class_counts = np.bincount(indexes)
    # Compute class weights
    class_weights = None
    if compute_class_weights:
        max_count = np.max(class_counts)
        class_weights = {c: max_count / class_counts[i] for i, c in enumerate(classes)}
    # Construct list of indexes for each unique class
    class_indexes = np.split(np.argsort(indexes, kind='mergesort'), np.cumsum(class_counts)[:-1])
    # Split indexes for each class into train and test
    train_indexes, test_indexes = [], []
    for class_index in class_indexes:
        assert len(class_index) > 1
        # Permutate indexes of current class
        class_index = np.random.permutation(class_index)
        # Choose place where to split indexes
        split = max(1, int(len(class_index) * test_fraction))
        # Split class indexes for test and train parts
        test_indexes.extend(class_index[:split])
        train_indexes.extend(class_index[split:])
    # Return permuted train and test indexes
    if shuffle:
        train_indexes = np.random.permutation(train_indexes)
        test_indexes = np.random.permutation(test_indexes)
    return (train_indexes, test_indexes, class_weights) if compute_class_weights else (train_indexes, test_indexes)


def k_fold_split(n: int,
                 k: int,
                 n_folds: int = 10,
                 n_parts: int = 1,
                 random_seed: Optional[Any] = None,
                 shuffle=True
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a number of samples `n` splits sequence [0...n] into train and test parts
    
    Notes
    -----
    
    Examples
    --------
    >>> t, v = k_fold_split(n=10, k=0, n_folds=10, shuffle=False)
    >>> print(t)  # Should print [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> print(v)  # Should print: [0]
    
    Parameters
    ----------
    n : int
        Number of samples in dataset
    k : int
        Specifies the number of fold to be used for test samples.
        0 - use the samples from first fold as test samples
        (n-1) - use the samples from last fold as test samples
    n_folds : int
        Number of folds to use.
        test_fraction will be: 1/n_folds
        train_fraction will be: (n_folds-1)/n_folds
    n_parts : int
        If more than 1 - split sequence [0...n] into `n_parts` equal parts.
        Then apply k-fold split to each part, and unite train/test samples from each part.
    random_seed : Optional[Any]
        Specify a seed value for `np.random.seed()`
        train and test indexes are randomly shuffled.
        You need to use the same seed value to get the same train/test indexes at different runs.
    shuffle : bool
        Specify True to shuffle train and test indices. Default: True
    
    Return
    ------
    (train_indexes, test_indexes): Tuple of two ndarrays
        train indexes and test indexes
    """
    assert (n > n_folds) and (n_parts > 0) and (n_folds > 0) and (0 <= k <= n_folds - 1)
    train_indexes = []
    test_indexes = []
    part_size = n / n_parts
    part_start = 0.0
    while (n - part_start) > 1:
        part_end = min(float(n), part_start + part_size)
        fold_size = (part_end - part_start) / n_folds
        if fold_size <= 0:
            continue
        i = 0
        fold_start = part_start
        while (part_end - fold_start) > 1:
            fold_end = min(part_end, fold_start + fold_size)
            fold_indices = np.arange(int(np.round(fold_start)), int(np.round(fold_end)))
            if k == i:
                test_indexes.append(fold_indices)
            else:
                train_indexes.append(fold_indices)
            i += 1
            fold_start = fold_end
        part_start = part_end
    train_indexes = np.concatenate(train_indexes)
    test_indexes = np.concatenate(test_indexes)
    assert len(train_indexes) + len(test_indexes) == n
    if random_seed is not None:
        np.random.seed(random_seed)
    if shuffle:
        train_indexes = np.random.permutation(train_indexes)
        test_indexes = np.random.permutation(test_indexes)
    return train_indexes, test_indexes


def k_fold_split_dates(date_from: date,
                       date_to: date,
                       k: int,
                       n_folds: int = 10,
                       n_parts: int = 1,
                       ) -> Tuple[List[Tuple[date, date]], List[Tuple[date, date]]]:
    """Given a number of samples `n` splits sequence [0...n] into train and test parts
    Parameters
    ----------
    date_from : date
        Starting date of a time period (included)
    date_to : date
        Ending date of a time period (included)
    k : int
        Specifies the number of fold to be used for test samples.
        0 - use the samples from first fold as test samples
        (n_folds-1) - use the samples from last fold as test samples
    n_folds : int
        Number of folds to use.
        test_fraction will be: 1/n_folds
        train_fraction will be: (n_folds-1)/n_folds
    n_parts : int
        If more than 1 - split sequence [0...n] into `n_parts` equal parts.
        Then apply k-fold split to each part, and unite train/test samples from each part.
    random_seed : Optional[Any]
        Specify a seed value for `np.random.seed()`
        train and test indexes are randomly shuffled.
        You need to use the same seed value to get the same train/test indexes at different runs.
    shuffle : bool
        Specify True to shuffle train and test indices. Default: True

    Return
    ------
    (train_indexes, test_indexes): Tuple of two ndarrays
        train indexes and test indexes
    """
    assert isinstance(date_from, date) and isinstance(date_to, date) and (date_from < date_to)
    days = (date_to - date_from).days + 1
    train_indexes, test_indexes = k_fold_split(n=days, k=k, n_folds=n_folds, n_parts=n_parts,
                                               random_seed=None, shuffle=False)
    
    def indices_to_periods(indices: Sequence[int]) -> List[Tuple[date, date]]:
        periods = []
        start_index, end_index = None, None
        for i, index in enumerate(indices):
            if (start_index is None) or (end_index is None):
                start_index = end_index = int(index)
            elif index - end_index == 1:
                end_index = int(index)
            elif index - end_index > 1:
                periods.append((date_from + timedelta(days=start_index), date_from + timedelta(days=end_index)))
                start_index = end_index = int(index)
            elif i == 0:
                pass
            else:
                raise RuntimeError(f'Invalid sequence of indices: {indices}')
        if end_index > 0:
            periods.append((date_from + timedelta(days=start_index), date_from + timedelta(days=end_index)))
        return periods

    train_periods = indices_to_periods(train_indexes)
    test_periods = indices_to_periods(test_indexes)
    return train_periods, test_periods
    

if __name__ == '__main__':
    y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    y = np.asarray(y)
    train, test, class_weights = stratified_split(y, 0.2, True)
    print(train)
    print(test)
    print(class_weights)

    k_fold_split(n=30, k=2, n_folds=3, n_parts=2)
    k_fold_split(n=31, k=2, n_folds=3, n_parts=2)

    train, test = k_fold_split_dates(
        date_from=date(2021, 3, 1),
        date_to=date(2021, 3, 31),
        k=2,
        n_folds=3,
        n_parts=2
    )
    print(train)
    print(test)
