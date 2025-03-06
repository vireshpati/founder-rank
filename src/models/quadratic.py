import torch
from torch import nn
from src.models.model_utils import initialize_weight_matrix


class QuadraticModel(nn.Module):
    def __init__(self, input_dim, rand_init=False):
        super(QuadraticModel, self).__init__()
        if rand_init:
            self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        else:
            W_init = torch.FloatTensor(initialize_weight_matrix(input_dim))
        self.W = nn.Parameter(W_init)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Enforce symmetry
        W_sym = 0.5 * (self.W + self.W.t())
        scores = torch.sum(x * (x @ W_sym), dim=1)
        return scores

    def get_W(self):
        return 0.5 * (self.W + self.W.t())


class QuadMLP(nn.Module):
    # Quadratic form captures pairwise explicitly and MLP for higher level interaction and non-linear pattern.
    #  Input -> Linear(dim->64) -> LayerNorm -> GELU -> Dropout(0.2)
    # -> Linear(64->32) -> LayerNorm -> GELU -> Dropout(0.1)
    # -> Linear(32->1)
    def __init__(self, input_dim, hidden_dim=64, rand_init=False):
        super().__init__()

        if rand_init:
            self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
        else:
            self.W = nn.Parameter(torch.FloatTensor(initialize_weight_matrix(input_dim)))

        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        W_sym = 0.5 * (self.W + self.W.t())
        quad_scores = torch.sum(x * (x @ W_sym), dim=1, keepdim=True)
        nonlinear_scores = self.nonlinear(x)
        return (quad_scores + nonlinear_scores + self.b).squeeze()

    def get_W(self):
        return 0.5 * (self.W + self.W.t())
