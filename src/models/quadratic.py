import torch
from torch import nn
from src.utils.model_utils import initialize_weight_matrix
import numpy as np


class QuadraticModel(nn.Module):
    def __init__(self, input_dim, rand_init=False):
        super(QuadraticModel, self).__init__()
      
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


class QuadMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        W_init = torch.FloatTensor(initialize_weight_matrix(input_dim))
        # Enforce  symmetry
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

