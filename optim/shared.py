import torch as th


class SharedRMSprop(th.optim.RMSprop):
    """
    Implements RMSprop algorithm with shared state (for multiprocess learning)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                         momentum=momentum, centered=centered)
        # Initialize optimizer state and make it shared
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = th.tensor(0, dtype=th.int, device=p.device)
                state['square_avg'] = th.zeros_like(p)
                if group['momentum'] > 0:
                    state['momentum_buffer'] = th.zeros_like(p)
                if group['centered']:
                    state['grad_avg'] = th.zeros_like(p)
                # Make all tensors shared
                for _, v in state.items():
                    if th.is_tensor(v):
                        v.share_memory_()


class SharedAdamW(th.optim.AdamW):
    """
    Implements AdamW algorithm with correct weight decay and shared state (for multiprocess learning)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        # Initialize optimizer state and make it shared
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = th.tensor(0, dtype=th.int, device=p.device)
                # Exponential moving average of gradient values
                state['exp_avg'] = th.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = th.zeros_like(p)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = th.zeros_like(p)
                # Make all tensors shared
                for _, v in state.items():
                    if th.is_tensor(v):
                        v.share_memory_()
