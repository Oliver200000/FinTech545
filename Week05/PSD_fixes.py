import numpy as np
import copy

# near_psd() method
def near_psd(a, epsilon=0.0):
    n = a.shape[1]

    inv_sd = None
    out = np.copy(a)

    # Calculate the correlation matrix if we got a covariance
    if np.allclose(np.diag(out), 1.0):
        inv_sd = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = inv_sd @ out @ inv_sd

    # SVD, update the eigenvalues and scale
    eigenvals, eigenvecs = np.linalg.eigh(out)
    eigenvals = np.maximum(eigenvals, epsilon)
    s = 1.0 / (np.square(eigenvecs) @ eigenvals)
    s = np.diagflat(np.sqrt(s))
    d = np.diag(np.sqrt(eigenvals))
    b = s @ eigenvecs @ d
    out = b @ b.T

    # Add back the variance
    if inv_sd is not None:
        inv_sd = np.diag(1.0 / np.diag(inv_sd))
        out = inv_sd @ out @ inv_sd

    return out


# Higham’s method
def frobenius_norm(matrix):
    return np.sqrt(np.square(matrix).sum())

def projection_u(matrix):
    out = np.copy(matrix)
    np.fill_diagonal(out, 1.0)
    return out

def projection_s(matrix, epsilon=0.0):
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    eigenvals = np.maximum(eigenvals, epsilon)
    return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

def higham_psd(a, max_iter=100, tol=1e-8):
    # delta_s0 = 0, Y0 = A, gamma0 = max float
    delta_s = 0.0
    y = a
    prev_gamma = np.inf

    # loop k iterations
    for i in range(max_iter):
        r = y - delta_s
        x = projection_s(r)
        delta_s = x - r
        y = projection_u(x)
        gamma = frobenius_norm(y - a)

        if abs(gamma - prev_gamma) < tol:  
            break

        prev_gamma = gamma

    return y





