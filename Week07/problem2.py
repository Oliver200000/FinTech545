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
import math
import sys
warnings.filterwarnings('ignore')

import inspect

# calculate first order derivative
def first_order_der(func, x, delta):
  return (func(x + delta) - func(x - delta)) / (2 * delta)

# calculate second order derivative
def second_order_der(func, x, delta):
  return (func(x + delta) + func(x - delta) - 2 * func(x)) / delta ** 2

def cal_partial_derivative(func, order, arg_name, delta=1e-3):
  # initialize for argument names and order
  arg_names = list(inspect.signature(func).parameters.keys())
  derivative_fs = {1: first_order_der, 2: second_order_der}

  def partial_derivative(*args, **kwargs):
    # parse argument names and order
    args_dict = dict(list(zip(arg_names, args)) + list(kwargs.items()))
    arg_val = args_dict.pop(arg_name)

    def partial_f(x):
      p_kwargs = {arg_name:x, **args_dict}
      return func(**p_kwargs)
    return derivative_fs[order](partial_f, arg_val, delta)
  return partial_derivative

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


all_prices = pd.read_csv("DailyPrices.csv", parse_dates=["Date"])
all_returns = calculate_returns(all_prices, method="discrete", date_col="Date")

div_date = datetime(2022, 3, 15)
current_date = datetime(2022, 3, 13)
r = 0.0025
coupon = 0.0053
b = r - coupon
N = 200

def n_nodes(N):
    return (N + 2) * (N + 1) // 2

def node_index(i, j):
    return n_nodes(j - 1) + i


def binomial_tree_no_div(option_type, S0, X, T, sigma, r, N):
  is_call = 1 if option_type == "Call" else -1
  dt = T / N
  disc = np.exp(-r * dt)
  u = np.exp(sigma * np.sqrt(dt))
  d = 1 / u
  p = (np.exp(r * dt) - d) / (u - d)
    
  C = np.empty(n_nodes(N), dtype=float)
            
  for i in np.arange(N, -1, -1):
    for j in range(i, -1, -1):
      S = S0 * u ** j * d ** (i - j)
      index = node_index(j, i)
      C[index] = max(0, (S - X) * is_call)
      if i < N:
        val = disc * (p * C[node_index(j + 1, i + 1)] + (1 - p) * C[node_index(j, i + 1)])
        C[index] = max(C[index], val)
                
  return C[0]

def binomial_tree(option_type, S0, X, T, div_time, div, sigma, r, N):
  if div_date is None or div is None:
    return binomial_tree_no_div(option_type, S0, X, T, sigma, r, N)
  
  is_call = 1 if option_type == "Call" else -1
  dt = T / N
  disc = np.exp(-r * dt)
    
  #calculate u, d, and p
  u = np.exp(sigma * np.sqrt(dt))
  d = 1 / u
  p = (np.exp(r * dt) - d) / (u - d)

  new_T = T - div_time * dt
  new_N = N - div_time

  C = np.empty(n_nodes(div_time), dtype=float)
  for i in range(div_time, -1, -1):
    for j in range(i, -1, -1):
      S = S0 * u ** j * d ** (i - j)
      val_exe = max(0, (S - X) * is_call)
      if i < div_time:
        val = disc * (p * C[node_index(j + 1, i + 1)] + (1 - p) * C[node_index(j, i + 1)])
      else:
        val = binomial_tree(option_type, S - div, X, new_T, None, None, sigma, r, new_N)
      C[node_index(j, i)] = max(val_exe, val)
    
  return C[0]

def implied_vol_american(option_type, S0, X, T, div_time, div, r, N, market_price, x0=0.5):
  def equation(sigma):
    return binomial_tree(option_type, S0, X, T, div_time, div, sigma, r, N) - market_price
  # Back solve the binomial tree valuation to get the implied volatility
  return scipy.optimize.fsolve(equation, x0=x0, xtol=0.00001)[0]

def calculate_sim_values(portfolios, sim_prices, days_ahead=0):
  sim_values = pd.DataFrame(index=portfolios.index, 
                            columns=list(range(sim_prices.shape[0])))
  sim_prices = np.array(sim_prices)
  for i in portfolios.index:
    if portfolios["Type"][i] == "Stock":
      # For stock, the single value is its price
      single_values = sim_prices
    else:
      # For option, calculate values with gbsm method
      option_type = portfolios["OptionType"][i]
      X = portfolios["Strike"][i]
      T = ((portfolios["ExpirationDate"][i] - current_date).days - days_ahead) / 365
      sigma = portfolios["ImpliedVol"][i]
      div_time = int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)
      div = 1
      option_values = []
      for S in sim_prices:
        option_values.append(binomial_tree(option_type, S, X, T, div_time, div, sigma, r, N))
      single_values = np.array(option_values)
    
    # Calculate the total values based on holding
    sim_values.loc[i, :] = portfolios["Holding"][i] * single_values
  
  # Combine the values for same portfolios
  sim_values['Portfolio'] = portfolios['Portfolio']
  return sim_values.groupby('Portfolio').sum()

S = 164.85
N = 25
current_date = datetime(2023, 3, 3)
div_date = datetime(2022, 3, 15)
r = 0.0425
div = 1
portfolios = pd.read_csv('problem2.csv', parse_dates=['ExpirationDate'])
portfolios['CurrentValue'] = portfolios['CurrentPrice'] * portfolios['Holding']
# Calculate the implied volatility for all portfolios
implied_vols = []
for i in range(len(portfolios.index)):
  if portfolios["Type"][i] == "Stock":
    implied_vols.append(None)
  else:
    option_type = portfolios["OptionType"][i]
    X = portfolios["Strike"][i]
    T = (portfolios["ExpirationDate"][i] - current_date).days / 365
    div_time = int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)
    market_price = portfolios["CurrentPrice"][i]
    sigma = implied_vol_american(option_type, S, X, T, div_time, div, r, N, market_price)
    implied_vols.append(sigma)

# Store the implied volatility in portfolios
portfolios["ImpliedVol"] = implied_vols
     

# Simulate the price in 120-220 range
sim_prices = np.linspace(120, 220, 20)

# Calculate the stock and option values
sim_values = calculate_sim_values(portfolios, sim_prices, 10)

# Plot the values for each portfolio
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
idx = 0
for portfolio, dataframe in sim_values.groupby('Portfolio'):
  i, j = idx // 3, idx % 3
  ax = axes[i][j]
  ax.plot(sim_prices, dataframe.iloc[0, :].values)
  ax.set_title(portfolio)
  ax.set_xlabel('Underlying Price', fontsize=8)
  ax.set_ylabel('Portfolio Value', fontsize=8)
  idx += 1

# Simulate the prices based on returns with normal distribution
std = all_returns['AAPL'].std()
np.random.seed(0)
sim_returns = scipy.stats.norm(0, std).rvs((10, 100))
sim_prices = 164.85 * (1 + sim_returns).prod(axis=0)

# Calculate the current values and sim values
portfolios["CurrentValue"] = portfolios["CurrentPrice"] * portfolios["Holding"]
curr_values = portfolios.groupby('Portfolio')['CurrentValue'].sum()
sim_values = calculate_sim_values(portfolios, sim_prices, 10)

# Calculate the value difference
sim_value_changes = (sim_values.T - curr_values).T

def calculate_var(data, mean=0, alpha=0.05):
  return mean - np.quantile(data, alpha)
def calculate_es(data, mean=0, alpha=0.05):
  var = calculate_var(data, mean, alpha)
  return -np.mean(data[data <= -var])
# Calculate the Mean, VaR and ES, and print the results
result = pd.DataFrame(index=sim_value_changes.index)
result['Mean'] = sim_value_changes.mean(axis=1)
result['VaR'] = sim_value_changes.apply(lambda x:calculate_var(x, 0), axis=1)
result['ES'] = sim_value_changes.apply(lambda x:calculate_es(x), axis=1)
print(result)



cal_amr_delta_num =  cal_partial_derivative(binomial_tree, 1, 'S0')

# Calculate the implied volatility for all portfolios
deltas = []
for i in range(len(portfolios.index)):
  if portfolios["Type"][i] == "Stock":
    deltas.append(1)
  else:
    option_type = portfolios["OptionType"][i]
    X = portfolios["Strike"][i]
    T = ((portfolios["ExpirationDate"][i] - current_date).days - 10) / 365
    div_time = int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)
    delta = cal_amr_delta_num(option_type, S, X, T, div_time, div, sigma, r, N)
    deltas.append(delta)

# Store the deltas in portfolios
portfolios["deltas"] = deltas

alpha = 0.05
t = 10
result_dn = pd.DataFrame(index=sorted(portfolios['Portfolio'].unique()), columns=['Mean', 'VaR', 'ES'])
result_dn.index.name = 'Portfolio'
for pfl, df in portfolios.groupby('Portfolio'):
  gradient = S / df['CurrentValue'].sum() * (df['Holding'] * df['deltas']).sum()
  pfl_10d_std = abs(gradient) * std * np.sqrt(t)
  N = scipy.stats.norm(0, 1)
  present_value = df['CurrentValue'].sum() 
  result_dn.loc[pfl]['Mean'] = 0
  result_dn.loc[pfl]['VaR'] = -present_value * N.ppf(alpha) * pfl_10d_std
  result_dn.loc[pfl]['ES'] = present_value * pfl_10d_std * N.pdf(N.ppf(alpha)) / alpha

print(result_dn)