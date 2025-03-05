import numpy as np
from ..config.config import cfg


def initialize_weight_matrix(K, MATRIX=cfg.MATRIX, seed=2):

    W = np.zeros((K, K))

    start_idx = 0
    for _, cfg in MATRIX.items():
        weight = cfg['WEIGHT']
        dim = cfg['DIMENSION']
        end_idx = start_idx + dim
        tiers = np.array(list(range(3, 3-dim,-1))[::-1]) * weight  
        indices = np.arange(start_idx, end_idx)
        W[indices, indices] = tiers
        start_idx = end_idx

    #enforce symmetry:
    return 0.5 * (W + W.T)

def score_feature_matrix(feature_matrix, W):

    return np.sum((feature_matrix @ W) * feature_matrix, axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_success_probability(x, W, b):
    # x should be a normalized feature vector.
    score = x.T @ W @ x + b
    return sigmoid(score)