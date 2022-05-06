from typing import Union, Tuple, Sequence, Callable, Optional, Literal
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


# TODO: Think about using nn.LayerNorm


class BaseModel(nn.Module, ABC):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: Union[int, Tuple[int, ...]],
                 activation_fn: Optional[Literal[
                     'relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'
                 ]] = None,
                 init: Optional[Literal['lecun', 'xavier', 'glorot', 'kaiming', 'he', 'orthogonal']] = None,
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        super().__init__()
        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
            self.num_channels = input_shape
            self.input_size = input_shape
        elif isinstance(input_shape, (Tuple, Sequence)):
            self.input_shape = tuple(input_shape)
            self.num_channels = input_shape[0]
            self.input_size: int = np.prod(input_shape).item()
        else:
            raise ValueError(f'Invalid input shape: {repr(input_shape)}')
        
        self.layers = layers
        
        self.activation_fn = activation_fn
        self.init = init
        self.dropout = dropout
        self.compute_layer_activity = compute_layer_activity
        
        self._mean_layer_activity: Optional[th.Tensor] = None
        self._mean_layer_activity_count = 0

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError()

    @property
    def state_size(self) -> int:
        return 0
    
    @property
    def mean_layer_activity(self):
        return self._mean_layer_activity / self._mean_layer_activity_count
    
    def reset(self):
        if self.compute_layer_activity:
            self._mean_layer_activity = None
            self._mean_layer_activity_count = 0
    
    def _update_mean_layer_activity(self, *args):
        for x in args:
            m = th.mean(x)
            if self._mean_layer_activity is None:
                self._mean_layer_activity = m
            else:
                self._mean_layer_activity += m
            self._mean_layer_activity_count += 1

    def forward(self, input: th.Tensor) -> th.Tensor:
        if input.dim() == 1 + len(self.input_shape):
            return self.step(input)

        # We have got input of shape (SeqLen, Batch, <input_shape>) - process all sequence at once
        assert (input.dim() == 2 + len(self.input_shape))
        
        # Process sequence step by step
        output = []
        for t in range(input.shape[0]):
            x = self.step(input[t])
            output.append(x)
            
        # Construct output tensor
        output = th.stack(output)
        return output
    
    def step(self, input: th.Tensor) -> th.Tensor:
        raise NotImplementedError()
    
    def extra_repr(self) -> str:
        return (
            f'activation_fn={self.activation_fn}\n'
            f'init={self.init}\n'
            f'dropout={self.dropout}\n'
            f'compute_layer_activity={self.compute_layer_activity}'
        )

    @staticmethod
    def init_weights(module: nn.Module, activation_fn: str, init: str):
        if init in {'kaiming', 'he'}:
            BaseModel.init_weights_kaiming(module, gain=activation_fn)
        elif init in {'xavier', 'glorot'}:
            BaseModel.init_weights_xavier(module, gain=activation_fn)
        elif init == 'orthogonal':
            BaseModel.init_weights_orthogonal(module, gain=1.0)
        else:
            raise ValueError(f'Unknown initialization scheme: {init}')

    @staticmethod
    def init_weights_kaiming(module: nn.Module, gain: Union[float, str] = 1.0):
        """Also known as He initialization"""
        if isinstance(gain, str):
            gain = BaseModel._get_gain_for_nonlinearity(gain)
    
        def _func(m):
            if type(m) == nn.Linear:
                fan = BaseModel._calculate_fan(m.weight, mode='fan_in')
                std = gain / math.sqrt(fan)
                with th.no_grad():
                    m.weight.normal_(0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
        module.apply(_func)

    @staticmethod
    def init_weights_xavier(module: nn.Module, gain: str):
        """Also known as Glorot initialization"""
        if isinstance(gain, str):
            gain = BaseModel._get_gain_for_nonlinearity(gain)
        
        def _func(m):
            if type(m) == nn.Linear:
                th.nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        module.apply(_func)

    @staticmethod
    def init_weights_orthogonal(module: nn.Module, gain: Union[float, str] = 1.0):
        if isinstance(gain, str):
            gain = BaseModel._get_gain_for_nonlinearity(gain)
    
        def _func(m):
            if type(m) == nn.Linear:
                th.nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        module.apply(_func)
    
    @staticmethod
    def _get_gain_for_nonlinearity(activation_fn: Literal[
                                      'relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'
                                   ]) -> float:
        if activation_fn == 'sigmoid':
            return 1
        elif activation_fn == 'tanh':
            return 5.0 / 3
        elif activation_fn == 'relu':
            return math.sqrt(2.0)
        elif activation_fn in {'leakyrelu', 'prelu'}:
            if activation_fn == 'leakyrelu':
                negative_slope = 0.01
            else:
                negative_slope = 0.25
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        elif activation_fn == 'elu':
            # TODO: Find real gain value
            return 3.0 / 4
        elif activation_fn == 'selu':
            return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
        else:
            raise ValueError(f'Unsupported activation_fn {activation_fn}')
    
    @staticmethod
    def _calculate_fan(tensor: th.Tensor,
                       mode: Literal['fan_in', 'fan_out', 'both'] = 'both'
                       ) -> Union[int, Tuple[int, int]]:
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError('Fan in and fan out can not be computed for tensor with fewer than 2 dimensions')
    
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
        return fan_in if (mode == 'fan_in') else fan_out if (mode == 'fan_out') else (fan_in, fan_out)

    @staticmethod
    def get_activation(name: str, num_features: int, as_module=False) -> Union[Callable, nn.Module]:
        if name == 'relu':
            return nn.ReLU(inplace=True) if as_module else F.relu
        elif name == 'elu':
            return nn.ELU(inplace=True) if as_module else F.elu
        elif name == 'prelu':
            if not as_module:
                raise ValueError('PReLU is a trainable activation layer. So it is only available as a module!')
            return nn.PReLU(num_features)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(inplace=True) if as_module else F.leaky_relu
        elif name == 'selu':
            return nn.SELU(inplace=True) if as_module else F.selu
        elif name == 'celu':
            return nn.CELU(inplace=True) if as_module else F.celu
        elif name == 'gelu':
            return nn.GELU() if as_module else F.gelu
        elif name == 'tanh':
            return nn.Tanh() if as_module else th.tanh
        elif name == 'sigmoid':
            return nn.Sigmoid() if as_module else th.sigmoid
        else:
            raise ValueError(f'Cant find activation function for: {name}')


class MLP(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: Union[int, Tuple[int, ...]],
                 activation_fn:
                    Literal['relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'] = 'tanh',
                 init: Optional[Literal['lecun', 'xavier', 'glorot', 'kaiming', 'he', 'orthogonal']] = 'he',
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 ** kwargs):
        if isinstance(layers, int):
            layers = [layers]
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=activation_fn, init=init,
                         dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        modules = []
        inp = self.input_size
        for i, out in enumerate(layers):
            modules.append(nn.Linear(in_features=inp, out_features=out, bias=True))
            if isinstance(dropout, float) and (0 < dropout < 1) and (0 <= i < len(layers)):
                modules.append(nn.AlphaDropout(p=dropout))
            modules.append(self.get_activation(activation_fn, out, as_module=True))
            inp = out
        self.net = nn.Sequential(*modules)
        if isinstance(init, str):
            self.init_weights(self.net, init=self.init, activation_fn=self.activation_fn)

    @property
    def output_size(self) -> int:
        return self.layers[-1]

    def step(self, x: th.Tensor) -> th.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        for module in self.net:
            x = module(x)
            if self.compute_layer_activity and (type(module) not in {nn.Linear, nn.AlphaDropout}):
                self._update_mean_layer_activity(x)
        return x


class ModelLSTM1(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: int,
                 activation_fn:
                    Literal['relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'] = 'tanh',
                 init: Optional[Literal['lecun', 'xavier', 'glorot', 'kaiming', 'he', 'orthogonal']] = 'he',
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        if 'dropout' in kwargs:
            del kwargs['dropout']
        if 'activation_fn' in kwargs:
            del kwargs['activation_fn']
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=activation_fn,
                         init=init, dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        self.lstm1 = nn.LSTMCell(input_size=self.input_size, hidden_size=layers, bias=True)
        self.state1: Optional[Tuple[th.Tensor, th.Tensor]] = None
        self.drop1 = nn.AlphaDropout(p=dropout) if isinstance(dropout, float) and (0 < dropout < 1) else None
        self.fc1 = nn.Linear(layers, layers, bias=True)
        self.act1 = self.get_activation(self.activation_fn, layers)
        if isinstance(init, str):
            self.init_weights(self.fc1, init=self.init, activation_fn=self.activation_fn)

    @property
    def output_size(self) -> int:
        return self.layers

    @property
    def state_size(self) -> int:
        return 2 * self.layers

    @property
    def state(self) -> Optional[th.Tensor]:
        if self.state1 is None:
            return None
        return th.cat(self.state1, dim=-1).detach()

    @state.setter
    def state(self, value: th.Tensor):
        assert (value.dim() == 2) and (value.shape[1] == self.layers * 2)
        self.state1 = (value[..., :self.layers], value[..., self.layers:])
        
    def reset(self):
        self.state1 = None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process input
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        self.state1 = self.lstm1(x, self.state1)
        self._update_mean_layer_activity(self.state1[0], self.state1[1])
        x = self.state1[0]
        if self.drop1:
            x = self.drop1(x)
        x = self.fc1(x)
        x = self.act1(x)
        self._update_mean_layer_activity(x)
        return x
    

class ModelLSTM2(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: Union[int, Tuple[int, int]],
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        if isinstance(layers, int):
            layers = (layers, layers)
        assert isinstance(layers, Sequence) and (len(layers) == 2), f'Invalid layers argument: {layers}'
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=None,
                         dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        hid1, hid2 = layers
        self.lstm1 = nn.LSTMCell(input_size=self.input_size, hidden_size=hid1, bias=True)
        self.drop1 = nn.AlphaDropout(p=dropout) if (dropout is not None) else None
        self.lstm2 = nn.LSTMCell(input_size=hid1, hidden_size=hid2, bias=True)
        self.state1: Optional[Tuple[th.Tensor, th.Tensor]] = None
        self.state2: Optional[Tuple[th.Tensor, th.Tensor]] = None

    @property
    def output_size(self) -> int:
        return self.lstm2.hidden_size

    @property
    def state_size(self) -> int:
        return 2 * self.layers[0] + 2 * self.layers[1]

    @property
    def state(self) -> Optional[Tuple[th.Tensor, th.Tensor]]:
        if (self.state1 is None) or (self.state2 is None):
            return None
        return th.cat((self.state1[0], self.state1[1], self.state2[0], self.state2[1]), dim=-1).detach()

    @state.setter
    def state(self, value: th.Tensor):
        hid1, hid2 = self.layers
        assert (value.dim() == 2) and (value.shape[1] == (2 * hid1 + 2 * hid2))
        self.state1 = (value[..., :hid1], value[..., hid1:(2*hid1)])
        self.state2 = (value[..., (2*hid1):(2*hid1 + hid2)], value[..., (2*hid1 + hid2):])
        
    def reset(self):
        self.state1, self.state2 = None, None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process input
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        self.state1 = self.lstm1(x, self.state1)
        self._update_mean_layer_activity(self.state1[0], self.state1[1])
        x = self.state1[0]
        if self.drop1:
            x = self.drop1(x)
        self.state2 = self.lstm2(x, self.state2)
        self._update_mean_layer_activity(self.state2[0], self.state2[1])
        x = self.state2[0]
        return x


class ModelGRU1(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: int,
                 compute_layer_activity=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=None,
                         dropout=None, compute_layer_activity=compute_layer_activity, **kwargs)
        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=layers, bias=True)
        self.state1: Optional[th.Tensor] = None

    @property
    def output_size(self) -> int:
        return self.layers

    @property
    def state_size(self) -> int:
        return self.layers

    @property
    def state(self) -> Optional[th.Tensor]:
        return self.state1.detach()

    @state.setter
    def state(self, value: th.Tensor):
        self.state1 = value

    def reset(self):
        self.state1 = None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process input
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        self.state1 = self.gru1(x, self.state1)
        x = self.state1
        self._update_mean_layer_activity(x)
        return x


class ModelGRU2(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: Union[int, Tuple[int, int]],
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=None,
                         dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        if isinstance(layers, int):
            hid1 = hid2 = layers
        elif isinstance(layers, Sequence) and (len(layers) == 2):
            hid1, hid2 = layers
        else:
            raise ValueError(f'Invalid layers argument: {layers}')
        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=hid1, bias=True)
        self.drop1 = nn.AlphaDropout(p=dropout) if (dropout is not None) else None
        self.gru2 = nn.GRUCell(input_size=hid1, hidden_size=hid2, bias=True)
        self.state1: Optional[th.Tensor] = None
        self.state2: Optional[th.Tensor] = None

    @property
    def output_size(self):
        return self.gru2.hidden_size

    @property
    def state_size(self) -> int:
        return self.layers[0] + self.layers[1]

    @property
    def state(self) -> Optional[Tuple[th.Tensor, th.Tensor]]:
        return self.state1.detach(), self.state2.detach()

    @state.setter
    def state(self, value: Tuple[th.Tensor, th.Tensor]):
        self.state1, self.state2 = value

    def reset(self):
        self.state1, self.state2 = None, None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process input
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        self.state1 = self.gru1(x, self.state1)
        x = self.state1
        self._update_mean_layer_activity(x)
        if self.drop1:
            x = self.drop1(x)
        self.state2 = self.gru2(x, self.state2)
        x = self.state2
        self._update_mean_layer_activity(x)
        return x


class ModelConvFC1(BaseModel):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: int = 128,
                 activation_fn:
                    Literal['relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'] = 'tanh',
                 init: Optional[Literal['lecun', 'xavier', 'glorot', 'kaiming', 'he', 'orthogonal']] = 'he',
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=activation_fn, init=init,
                         dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        assert isinstance(input_shape, Sequence) and (len(input_shape) == 3), f'Invalid input_shape: {input_shape}'
        Ci, Hi, Wi = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=Ci, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        Co, Ho, Wo = 64, (Hi // 32), (Wi // 32)
        self.drop1 = nn.AlphaDropout(p=dropout) if (dropout is not None) else None
        self.fc1 = nn.Linear(in_features=(Co * Ho * Wo), out_features=layers, bias=True)
        self.act1 = self.get_activation(activation_fn, layers)
        if isinstance(init, str):
            self.init_weights(self.fc1, self.init, self.activation_fn)

    @property
    def output_size(self) -> int:
        return self.layers

    def step(self, x: th.Tensor) -> th.Tensor:
        # Process convolution
        x = self.conv(x)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if self.drop1:
            x = self.drop1(x)
        x = self.fc1(x)
        x = self.act1(x)
        self._update_mean_layer_activity(x)
        return x


class ModelConvGRU1(BaseModel):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 layers: int = 128,
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=None,
                         dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        assert isinstance(input_shape, Sequence) and (len(input_shape) == 3), f'Invalid input_shape: {input_shape}'
        Ci, Hi, Wi = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=Ci, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
        )
        Co, Ho, Wo = 64, (Hi // 32), (Wi // 32)
        self.drop1 = nn.AlphaDropout(p=dropout) if (dropout is not None) else None
        self.gru1 = nn.GRUCell(input_size=(Co * Ho * Wo), hidden_size=layers, bias=True)
        self.state1: Optional[th.Tensor] = None

    @property
    def output_size(self) -> int:
        return self.layers

    @property
    def state_size(self) -> int:
        return self.layers

    @property
    def state(self) -> Optional[th.Tensor]:
        return self.state1.detach()

    @state.setter
    def state(self, value: th.Tensor):
        self.state1 = value

    def reset(self):
        self.state1 = None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process convolution
        x = self.conv(x)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        if self.drop1:
            x = self.drop1(x)
        self.state1 = self.gru1(x, self.state1)
        x = self.state1
        self._update_mean_layer_activity(x)
        return x


class ModelAttention(BaseModel):
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 layers: Union[int, Tuple[int, int]],
                 maxlen: int = 20,
                 activation_fn:
                    Literal['relu', 'elu', 'prelu', 'leakyrelu', 'selu', 'celu', 'gelu', 'tanh', 'sigmoid'] = 'tanh',
                 init: Optional[Literal['lecun', 'xavier', 'glorot', 'kaiming', 'he', 'orthogonal']] = 'he',
                 dropout: Optional[float] = None,
                 compute_layer_activity=False,
                 **kwargs):
        if 'dropout' in kwargs:
            del kwargs['dropout']
        if 'activation_fn' in kwargs:
            del kwargs['activation_fn']
        super().__init__(input_shape=input_shape, layers=layers, activation_fn=activation_fn,
            init=init, dropout=dropout, compute_layer_activity=compute_layer_activity, **kwargs)
        self.maxlen = maxlen
        if isinstance(layers, Tuple):
            assert len(layers) == 2
            key_dim, val_dim = layers
        elif isinstance(layers, int):
            key_dim, val_dim = layers, layers
        else:
            raise ValueError(f'Invalid layers argument: {layers}')
        assert isinstance(key_dim, int) and (key_dim > 0)
        assert isinstance(val_dim, int) and (val_dim > 0)
        self.key_dim, self.val_dim = key_dim, val_dim
        self.drop1 = nn.AlphaDropout(p=dropout) if isinstance(dropout, float) and (0 < dropout < 1) else None
        self.K = nn.Sequential(
            nn.Linear(self.input_size, key_dim, bias=True),
            self.get_activation(self.activation_fn, key_dim)
        )
        self.Q = nn.Sequential(
            nn.Linear(self.input_size, key_dim, bias=True),
            self.get_activation(self.activation_fn, key_dim)
        )
        self.V = nn.Sequential(
            nn.Linear(self.input_size, val_dim, bias=True),
            self.get_activation(self.activation_fn, val_dim)
        )
        self.act1 = self.get_activation(self.activation_fn, layers)
        if isinstance(init, str):
            self.init_weights(self.fc1, init=self.init, activation_fn=self.activation_fn)
    
    @property
    def output_size(self) -> int:
        return self.val_dim
    
    @property
    def state_size(self) -> int:
        return 2 * self.layers
    
    @property
    def state(self) -> Optional[th.Tensor]:
        if self.state1 is None:
            return None
        return th.cat(self.state1, dim=-1).detach()
    
    @state.setter
    def state(self, value: th.Tensor):
        assert (value.dim() == 2) and (value.shape[1] == self.layers * 2)
        self.state1 = (value[..., :self.layers], value[..., self.layers:])
    
    def reset(self):
        self.state1 = None
    
    def step(self, x: th.Tensor) -> th.Tensor:
        # Process input
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        self.state1 = self.lstm1(x, self.state1)
        self._update_mean_layer_activity(self.state1[0], self.state1[1])
        x = self.state1[0]
        if self.drop1:
            x = self.drop1(x)
        x = self.fc1(x)
        x = self.act1(x)
        self._update_mean_layer_activity(x)
        return x


Models = {
    'mlp': MLP,
    'lstm1': ModelLSTM1,
    'lstm2': ModelLSTM2,
    'gru1': ModelGRU1,
    'gru2': ModelGRU2,
    'convfc1': ModelConvFC1,
    'convgru1': ModelConvGRU1,
}
