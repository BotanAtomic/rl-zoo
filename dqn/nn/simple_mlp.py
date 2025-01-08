import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
import torchinfo


class SimpleMLP(nn.Module):

    def __init__(self, space_dim: tuple[int, ...], action_dim: int, n_hidden_layers = 2, hidden_dim = 32):
        super().__init__()
        self.space_dim = np.array(space_dim).prod()
        self.action_dim = action_dim

        self.n_hidden_layers = n_hidden_layers
        self.hidden = hidden_dim

        self.model = self._build_model()
        torchinfo.summary(self.model, input_size=(1, self.space_dim))

    def _build_model(self):
        layers = [
            nn.Linear(self.space_dim, self.hidden),
            nn.ReLU()
        ]
        for _ in range(self.n_hidden_layers):
            layers.append(nn.Linear(self.hidden, self.hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden, self.action_dim))
        model = Sequential(*layers)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
