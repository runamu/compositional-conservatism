import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None,
        layernorm: Optional[bool] = False,
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]
            if layernorm:
                model += [nn.LayerNorm(out_dim)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            if layernorm: # is it okay if the output is normalized? # actor의 경우 dist net이 있어서 괜찮음 # critic도 self.last가 있어서 괜찮음
                model += [nn.LayerNorm(output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def pytorch_init(fan_in: float):
    """
    Custom initializer for PyTorch Linear layer weights and biases, mimicking the original JAX implementation.
    """
    bound = math.sqrt(1 / fan_in)
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -bound, bound)
            nn.init.constant_(m.bias, 0.1)  # Initialize biases to 0.1
    return _init

def uniform_init(bound: float):
    """
    Uniform initializer with a specified bound.
    """
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -bound, bound)
            nn.init.uniform_(m.bias, -bound, bound)
    return _init