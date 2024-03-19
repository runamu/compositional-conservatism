import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets import EnsembleLinear

class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        num_ensemble: int = 10,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [EnsembleLinear(in_dim, out_dim, num_ensemble), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [EnsembleLinear(hidden_dims[-1], output_dim, num_ensemble)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

        self._num_ensemble = num_ensemble

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)