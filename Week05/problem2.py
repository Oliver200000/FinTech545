import pandas as pd
import numpy as np
from ES_calculation import historical_ES
from PSD_fixes import higham_psd,near_psd
from VaR import readfile,Calculate_his,Calculate_norm
from Covariance import calculate_ewcov
from Simulation import direct_simulation,pca_simulation

# Load data from CSV
data = pd.read_csv('/Users/oliver/Desktop/FinTech 545/Week05/Project/problem1.csv')
returns = readfile('/Users/oliver/Desktop/FinTech 545/Week05/Project/DailyReturn.csv')


# Check Covariance estimation techniques when decay = 0.5
covariance = calculate_ewcov(returns, 0.5)
print("Covariance matrix")
print(covariance)

# Check Non PSD fixes for correlation matrices
n = 500
sigma = np.matrix(np.full((n, n), 0.9))
np.fill_diagonal(sigma, 1)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357


def check_psd(matrix, tol=1e-7):
    eigenvals, _ = np.linalg.eigh(matrix)
    return np.allclose(eigenvals, np.maximum(eigenvals, 0), rtol=0, atol=tol)
# near_psd() 
near_psd_matrix = near_psd(sigma)
if check_psd(near_psd_matrix):
    print("near_psd() method is correct")
# Higham's method
higham_psd_matrix = higham_psd(sigma)
if check_psd(higham_psd_matrix):
    print("Higham’s method is correct")

# Check Simulations
# Check Direct Simulation
print("Direct Simulation Matrix")
print(direct_simulation(covariance))
# Check PCA Simulation
print("PCA Simulation Matrix")
print(pca_simulation(covariance,1))

# Check Var calculation methods
# Historic method
print("VaR For Historic method:")
print(Calculate_his(returns))
print("VaR For Normal Distribution:")
# For Normal Distribution
print(Calculate_norm(returns))

# Check ES calculation
print("ES calculation is "+str(historical_ES(data, alpha=0.05)))
