from Library import calculate_ewcov, historical_ES,readfile,Calculate_his,Calculate_norm
import pandas as pd
from scipy.stats import norm, t
import numpy as np

def calculate_returns(price_series, method='arithmetic'):
    changes = price_series.pct_change()[1:]
    if method == 'arithmetic':
        returns = changes
    elif method == 'log':
        returns = np.log(1 +changes)

    return returns

# load in data and calculate returns
prices = pd.read_csv("/Users/oliver/Desktop/FinTech 545/Week05/Project/DailyPrices.csv", parse_dates=[0], index_col=0)
portfolios = pd.read_csv("/Users/oliver/Desktop/FinTech 545/Week05/Project/portfolio.csv")
returns = calculate_returns(prices)

total = portfolios.groupby('Stock').sum('Holding')
total['Portfolio'] = 'Total'
total = total.reset_index()
portfolios = portfolios.append(total)

t_params = {}
sim_data = []
for col in returns:
  stock_returns = returns[col]

  stock_returns -= stock_returns.mean()
  result = t.fit(stock_returns, method="MLE")
  df, loc, scale = result
  t_params[col] = [df, loc, scale]
  sim_data.append(t(df, loc, scale).rvs(10000))
sim_data = np.array(sim_data)

simulated_returns = pd.DataFrame(columns=returns.columns, data=sim_data.T)


current_prices = pd.DataFrame({"Price":prices.iloc[-1]})

#Fit a Generalized T model 

for name, portfolio in portfolios.groupby('Portfolio'):
    portfolio = portfolio.set_index('Stock')
    portfolio = portfolio.join(current_prices.loc[portfolio.index])

    sim_returns = simulated_returns[portfolio.index]
    sim_prices_change = sim_returns * portfolio['Price'].T
    sim_values_change = sim_prices_change @ portfolio['Holding']

    var = Calculate_his(sim_values_change)
    es = historical_ES(sim_values_change)
    print(f" {name} " + "VaR: " + str(var))
    print(f" {name} " + "ES: " + str(es))
    print()
     

