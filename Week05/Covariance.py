import numpy as np

# Calculate exponential weights
def calculate_exponential_weights(num_lags, decay):
    weights = []
    for i in range(1, num_lags + 1):
        weight = (1 - decay) * decay ** (i - 1)
        weights.append(weight)
    weights = np.array(weights)
    normalized_weights = weights / weights.sum()
    return normalized_weights

# Calculate exponentially weighted covariance matrix
def calculate_ewcov(data, decay):
    weights = calculate_exponential_weights(data.shape[1], decay)
    error_matrix = data - data.mean(axis=1)[:, np.newaxis]
    ewcov = error_matrix @ np.diag(weights) @ error_matrix.T
    return ewcov