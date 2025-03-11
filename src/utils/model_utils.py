import numpy as np
from ..config.config import cfg


def initialize_weight_matrix(
    K,
    MATRIX=cfg.MATRIX,
    seed=2,
    eps=0.0,
):
    np.random.seed(seed)
    W = np.zeros((K, K))

    start_idx = 0
    for _, cfg in MATRIX.items():
        weight = cfg["WEIGHT"]
        dim = cfg["DIMENSION"]
        end_idx = start_idx + dim
        tiers = np.array(list(range(3, 3 - dim, -1))[::-1]) * weight
        indices = np.arange(start_idx, end_idx)
        W[indices, indices] = tiers
        start_idx = end_idx

    # Add random noise
    if eps > 0:
        W += np.random.normal(0, eps, size=W.shape)

    # enforce symmetry:
    return 0.5 * (W + W.T)


def score_feature_matrix(feature_matrix, W):
    return np.sum((feature_matrix @ W) * feature_matrix, axis=1)