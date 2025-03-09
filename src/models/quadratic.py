import torch
from torch import nn
from src.utils.model_utils import initialize_weight_matrix
import numpy as np


class QuadraticModel(nn.Module):
    def __init__(self, input_dim, rand_init=False):
        super(QuadraticModel, self).__init__()
        # Add scaling factor
        self.quad_scale = nn.Parameter(torch.tensor(0.1))
        
        if rand_init:
            self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        else:
            W_init = torch.FloatTensor(initialize_weight_matrix(input_dim))
            self.W = nn.Parameter(W_init)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Enforce symmetry
        W_sym = 0.5 * (self.W + self.W.t())
        scores = torch.sum(x * (x @ W_sym), dim=1) * self.quad_scale + self.b
        return scores

    def get_W(self):
        return 0.5 * (self.W + self.W.t())

    def backward_hook(self, grad):
        # Scale down diagonal gradients
        grad.diagonal().div_(10.0)
        return grad

    def __init__(self, input_dim, rand_init=False):
        super(QuadraticModel, self).__init__()
        # Add scaling factor
        self.quad_scale = nn.Parameter(torch.tensor(0.1))
        
        if rand_init:
            self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        else:
            W_init = torch.FloatTensor(initialize_weight_matrix(input_dim))
            self.W = nn.Parameter(W_init)
        self.b = nn.Parameter(torch.zeros(1))
        self.W.register_hook(self.backward_hook)


class QuadMLP(nn.Module):
    # Quadratic form captures pairwise explicitly and MLP for higher level interaction and non-linear pattern.
    #  Input -> Linear(dim->64) -> LayerNorm -> GELU -> Dropout(0.2)
    # -> Linear(64->32) -> LayerNorm -> GELU -> Dropout(0.1)
    # -> Linear(32->1)
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        W_init = torch.FloatTensor(initialize_weight_matrix(input_dim))
        # Enforce initial symmetry
        W_init = 0.5 * (W_init + W_init.T)
        self.W = nn.Parameter(W_init)
        self.b = nn.Parameter(torch.zeros(1))
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        W_sym = 0.5 * (self.W + self.W.t())
        quad = torch.sum(x * (x @ W_sym), dim=1)
        mlp = self.mlp(x).squeeze()
        return quad + mlp + self.b

    def get_W(self):
        return 0.5 * (self.W + self.W.t())

    def backward_hook(self, grad):
        # Scale down diagonal gradients
        grad.diagonal().div_(10.0)
        return grad
