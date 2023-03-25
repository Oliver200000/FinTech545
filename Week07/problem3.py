# import packages
import math
import sys
from scipy.stats import norm, t
import scipy
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

def calculate_returns(data, method="discrete", date_col="Date"):
    assert date_col in data.columns

    price_vars = list(data.columns)
    price_vars.remove(date_col)
    price_data = data[price_vars]

    num_vars = len(price_vars)
    num_rows = len(price_data) - 1

    returns_data = np.zeros(shape=(num_rows, num_vars))
    returns_df = pd.DataFrame(returns_data)

    for i in range(num_rows):
        for j in range(num_vars):
            returns_df.iloc[i, j] = price_data.iloc[i + 1, j] / price_data.iloc[i, j]
            if method == "discrete":
                returns_df.iloc[i, j] -= 1
            elif method == "log":
                returns_df.iloc[i, j] = math.log(returns_df.iloc[i, j])
            else:
                sys.exit(1)

    date_series = data[date_col].drop(index=0)
    date_series.index -= 1

    output = pd.DataFrame({date_col: date_series})

    for i in range(num_vars):
        output[price_vars[i]] = returns_df.iloc[:, i]

    return output

# data preparation
ff = pd.read_csv('F-F_Research_Data_Factors_daily.csv', parse_dates=['Date']).set_index('Date')
mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', parse_dates=['Date']).set_index('Date')
# transfer percentage to value
data = ff.join(mom, how='right') / 100

all_prices = pd.read_csv("DailyPrices.csv", parse_dates=["Date"])
all_returns = calculate_returns(all_prices, method="discrete", date_col="Date")

stocks = ['AAPL', 'META', 'UNH', 'MA',  
          'MSFT' ,'NVDA', 'HD', 'PFE',  
          'AMZN' ,'BRK-B', 'PG', 'XOM',  
          'TSLA' ,'JPM' ,'V', 'DIS',  
          'GOOGL', 'JNJ', 'BAC', 'CSCO']
factors = ['Mkt-RF', 'SMB', 'HML', 'RF']
dataset = all_returns[stocks].join(data)

# calculate arithmetic E(r) in past 10 years
avg_factor_rets = data.loc['2012-1-14':'2022-1-14'].mean(axis=0)
avg_daily_rets = pd.Series()
for stock in stocks:
  # calculate betas
  model = sm.OLS(dataset[stock] - dataset['RF'], sm.add_constant(dataset[factors]))
  results = model.fit()
    
  # assume alpha = 0
  avg_daily_rets[stock] = (results.params[factors] * avg_factor_rets[factors]).sum() \
                          + avg_factor_rets['RF'] 
# geometric annual returns: mean and covariance
geo_means = np.log(1 + avg_daily_rets) * 255  
geo_covariance = np.log(1 + all_returns[stocks]).cov() * 255
print(geo_means)

print(geo_covariance)


# arithmetic annual returns: mean and covariance
arith_means = np.exp(geo_means + np.diagonal(geo_covariance.values) / 2) - 1

nstocks = geo_covariance.shape[0]
arith_covariance = np.empty((nstocks, nstocks), dtype=float)
for i in range(nstocks):
  for j in range(i, nstocks):
    mu_i, mu_j = geo_means.iloc[i], geo_means.iloc[j]
    sigma2_i, sigma2_j = geo_covariance.iloc[i, i], geo_covariance.iloc[j, j]
    sigma_ij = geo_covariance.iloc[i, j]
    arith_covariance[i, j] = np.exp(mu_i + mu_j + (sigma2_i + sigma2_j) / 2) * (np.exp(sigma_ij) - 1)
    arith_covariance[j, i] = arith_covariance[i, j]
arith_covariance = pd.DataFrame(arith_covariance, columns=stocks, index=stocks)

print(arith_means)


# calculate the most efficient portfolio which has the highest Sharpe ratio 
def neg_sharpe_ratio(weights, mean, cov, r):
  returns = mean @ weights.T
  std = np.sqrt(weights @ cov @ weights.T)
  return -(returns - r) / std

args = (arith_means, arith_covariance, 0.0025)
bounds = [(0.0, 1) for _ in stocks]
x0 = np.array(nstocks*[1 / nstocks])
constraints = {'type':'eq', 'fun': lambda x: np.sum(x) - 1}
results = scipy.optimize.minimize(neg_sharpe_ratio, x0=x0, args=args, bounds=bounds, constraints=constraints)
opt_sharpe, opt_weights = -results.fun, pd.Series(results.x, index=stocks)
opt_weights = pd.DataFrame(opt_weights, columns=['weights(%)'])
opt_weights['weights(%)'] = round(opt_weights*100, 2)

print("The most efficient portfolio consists of: ")
display(opt_weights)
print("The Portfolio's Sharpe Ratio is: " + str(opt_sharpe))