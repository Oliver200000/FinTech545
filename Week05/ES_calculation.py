import pandas as pd
import numpy as np
import scipy.stats as stats



# ES calculation
def historical_ES(x, alpha=0.05):
    if not 0 <= alpha <= 1:
        raise ValueError("Input 'alpha' must be between 0 and 1.")
    VaR = np.quantile(x, alpha)
    ES = -x[x <= VaR].mean().item()
    return ES


