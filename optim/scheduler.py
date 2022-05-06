from typing import Literal, Callable, Union, Tuple


class Scheduler(object):
    def __init__(self,
                 value: Union[float, Tuple[float, float]],
                 kind: Literal['linear', 'l', 's_shape', 's', 'constant', 'c'] = 'linear'):
        self.initial = value[0] if isinstance(value, Tuple) else value
        self.final = value[1] if isinstance(value, Tuple) else 0.0
        self.kind = kind
    
    def __call__(self, progress: float):
        if self.kind in {'linear', 'l'}:
            k = min(1.0, max(0.0, progress))
            return self.initial * (1 - k) + self.final * k
        elif self.kind in {'s_shape', 's'}:
            k = s_curve(progress)
            return self.initial * (1 - k) + self.final * k
        return self.initial
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.kind}: {self.initial} -> {self.final})'


def get(lr: Union[float, Tuple[float, float]],
        kind: Literal['linear', 'l', 's_shape', 's', 'constant', 'c'] = 'linear',
        ) -> Union[float, Callable[[float], float]]:
    
    initial_lr = lr[0] if isinstance(lr, Tuple) else lr
    final_lr = lr[1] if isinstance(lr, Tuple) else 0.0

    def linear(progress: float):
        k = min(1.0, max(0.0, progress))
        return initial_lr * (1 - k) + final_lr * k

    def s_shape(progress: float):
        k = s_curve(progress)
        return initial_lr * (1 - k) + final_lr * k
    
    if kind in {'linear', 'l'}:
        return linear
    elif kind in {'s_shape', 's'}:
        return s_shape
    assert isinstance(lr, float)
    return lr


def s_curve(x: float, k: float = 1.5) -> float:
    """
    S-shaped curve with adjustable curvature

    Notes
    -----

    ```
    1|            _--¯¯¯¯
     |          ∕
     |        ∕
    0| ___--¯
     ------------------------
      0                  1
    ```

    S-shaped curve has following properties:

    - when x: 0 -> 1, the value of function f(x) smoothly changes: 0 -> 1
    - x<=0:  f(x)=0
    - x=0.5: f(x)=0.5
    - x>=1:  f(x)=1
    - s-shape, with curvature depending on `k` argument

    When k=1 you get a straight line (0,0) - (1,1).
    When k>1 you get a smooth s-shaped curve.

    Taken from here:
    https://stats.stackexchange.com/questions/214877/is-there-a-formula-for-an-s-shaped-curve-with-domain-and-range-0-1

    Parameters
    ----------
    x : float
        Input value to compute s-curve.
        Can be any value from (-∞, +∞), but, generally, only interval [0, 1] has some meaning.
    k : float
        Curvature coefficient [1, ∞]
        Choose 1 - for a straight line.
        Choose 1.5 ... 2.0 - for an s-shaped curved line.
        Default: 1.5

    Returns
    -------
    value : float
    """
    assert (k >= 1)
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    return 1 / (1 + (x / (1 - x)) ** (-k))
