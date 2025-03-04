import numpy as np
from src.config import cfg


def initialize_weight_matrix(K, MATRIX=cfg.MATRIX, seed=42):

    W = np.zeros((K, K))

    start_idx = 0
    for cat, cfg in MATRIX.items():
        weight = cfg['WEIGHT']
        dim = cfg['DIMENSION']
        end_idx = start_idx + dim
        tiers = np.array(list(range(3, 3-dim,-1))[::-1]) * weight  
        indices = np.arange(start_idx, end_idx)
        W[indices, indices] = tiers
        start_idx = end_idx

    #Add small random noise to off-diagonal elements
    np.random.seed(seed) 
    noise = np.random.normal(0, 0.25, (K, K))
    np.fill_diagonal(noise, 0)
    W += noise

    #enforce symmetry:
    W = 0.5 * (W + W.T)
    
    return W

def score_feature_matrix(feature_matrix, W):

    return np.sum((feature_matrix @ W) * feature_matrix, axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_success_probability(x, W, b):
    # x should be a normalized feature vector.
    score = x.T @ W @ x + b
    return sigmoid(score)