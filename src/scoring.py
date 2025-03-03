import numpy as np

def one_hot_encode_column(values, dimension):
    if dimension == 3:
        indices = values - 1
    else:
        indices = values
    indices = np.clip(indices, 0, dimension - 1)
    return np.eye(dimension, dtype=int)[indices]