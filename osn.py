import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectScaleNet(nn.Module):
    def __init__(self, input_dim, n_dim, n_layer=2, alpha=1., shift=1.):
        super().__init__()
        self.alpha = alpha
        self.shift = shift

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, n_dim))
        for _ in range(1, n_layer):
            self.layers.append(nn.Linear(n_dim, n_dim))
        self.layers.append(nn.Linear(n_dim, 1, bias=False))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        prob = F.sigmoid(self.alpha * x + self.shift)
        return prob