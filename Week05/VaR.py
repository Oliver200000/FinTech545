import numpy as np
import pandas as pd


# File Reader
def readfile(name):
    data = np.genfromtxt(name, delimiter=',')
    data = np.delete(data, 0, 0)
    data = np.delete(data, 0, 1)
    return data



def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)

# Calculate historic VaR.
def Calculate_his(data):
    var_hist = calculate_var(data)
    return(var_hist)


# Calculate for VaR normal distribution
def Calculate_norm(data):
    sigma = np.std(data)
    simulation_norm = np.random.normal(0, sigma, 10000)
    var_norm = calculate_var(simulation_norm)
    return(var_norm)
     

     