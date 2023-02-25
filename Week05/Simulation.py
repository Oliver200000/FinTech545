import numpy as np
import scipy
from bisect import bisect_left

def chol_psd(L, A):
    n = A.shape[0]
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(L[j, :j], L[j, :j].T)
        temp = A[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        L[j, j] = np.sqrt(temp)
        if L[j, j] == 0.0:
            L[j, j:n-1] = 0.0
        else:
            for i in range(j+1, n):
                s = np.dot(L[i, :j], L[j, :j].T)
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L

def direct_simulation(cov, n_samples=25000):
    L = np.zeros_like(cov)
    L = chol_psd(L, cov)
    Z = scipy.random.randn(L.shape[0], n_samples)
    return L @ Z

def pca_simulation(cov, pct_explained, n_samples=25000):
    eigen_vals, eigen_vecs = np.linalg.eigh(cov)

    # Calculate PCA cumulative explained variance ratio (EVR)
    sorted_index = np.argsort(eigen_vals)[::-1]
    sorted_eigenvals = eigen_vals[sorted_index]
    sorted_eigenvectors = eigen_vecs[:, sorted_index]

    evr = sorted_eigenvals / sorted_eigenvals.sum()
    cumulative_evr = evr.cumsum()
    cumulative_evr[-1] = 1

    # Find the index for each explained variance threshold
    idx = bisect_left(cumulative_evr, pct_explained)

    explained_vals = np.clip(sorted_eigenvals[:idx + 1], 0, np.inf)
    explained_vecs = sorted_eigenvectors[:, :idx + 1]

    L = explained_vecs @ np.diag(np.sqrt(explained_vals))
    Z = scipy.random.randn(L.shape[1], n_samples)
    return L @ Z