import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import fsolve
from scipy.stats import norm

df = pd.read_csv('problem3.csv')
portfolios = df['Portfolio'].unique()

S = [50, 100, 150, 200, 250]
r = 0.0425  # Risk-free rate
q = 0.0053  # Dividend rate
b = r - q
underlying = 151.03

def gbsm(option_type, stock_price, strike, T, rf, b, ivol):
    d1 = (np.log(stock_price / strike) + (b + ivol**2 / 2) * T) / (ivol * np.sqrt(T))
    d2 = d1 - ivol * np.sqrt(T)
    if option_type == 'Call':
        return stock_price * np.exp((b - rf) * T) * norm.cdf(d1) - strike * np.exp(-rf * T) * norm.cdf(d2)
    else:
        return strike * np.exp(-rf * T) * norm.cdf(-d2) - stock_price * np.exp((b - rf) * T) * norm.cdf(-d1)

def implied_volatility(option_type, stock_price, strike, T, rf, b, market_price):
    f = lambda ivol: gbsm(option_type, stock_price, strike, T, rf, b, ivol) - market_price
    return fsolve(f, 0.5)

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'])
df['TTM'] = (df['ExpirationDate'] - datetime(2023, 3, 3)).dt.days / 365
df['ImpliedVol'] = df.apply(lambda row: implied_volatility(row['OptionType'], underlying, row['Strike'], row['TTM'], r, b, row['CurrentPrice']) if row['Type'] == 'Option' else 0, axis=1)

def portfolio_value(portfolio, stock_price):
    portfolio_df = df[df['Portfolio'] == portfolio]
    portfolio_value = 0
    for _, row in portfolio_df.iterrows():
        if row['Type'] == 'Stock':
            portfolio_value += row['Holding'] * stock_price
        else:
            option_value = gbsm(row['OptionType'], stock_price, row['Strike'], row['TTM'], r, b, row['ImpliedVol'])
            portfolio_value += row['Holding'] * option_value
    return portfolio_value

for portfolio in portfolios:
    plt.figure(figsize=(5, 5))
    portfolio_values = [portfolio_value(portfolio, stock_price) for stock_price in S]
    plt.plot(S, portfolio_values)
    plt.title(portfolio)
    plt.xlabel('Stock Price')
    plt.ylabel('Portfolio Value')
    plt.grid()
    plt.show()

daily_price = pd.read_csv('DailyPrices.csv')
aapl = daily_price['AAPL']
lreturn = np.diff(np.log(aapl))
aapl_return = lreturn - lreturn.mean()

model = sm.tsa.ARIMA(aapl_return, order=(1, 0, 0))
result = model.fit()
summary = result.summary()
forecast = result.forecast(steps=10)

m = float(summary.tables[1].data[1][1])
a1 = float(summary.tables[1].data[2][1])
s = math.sqrt(float(summary.tables[1].data[3][1]))

num_simulations = 10000
num_days = 10

sim = pd.DataFrame(0, index=range(num_simulations), columns=[f"Day {i+1}" for i in range(num_days)])

for i in range(num_days):
    for j in range(num_simulations):
        if i == 0:
            sim.iloc[j, i] = a1 * aapl_return.iloc[-1] + s * np.random.normal() + m
        else:
            sim.iloc[j, i] = a1 * sim.iloc[j, i-1] + s * np.random.normal() + m

sim_p = pd.DataFrame(0, index=range(num_simulations), columns=[f"Day {i+1}" for i in range(num_days)])

for i in range(num_days):
    if i == 0:
        sim_p.iloc[:, i] = np.exp(sim.iloc[:, i]) * underlying
    else:
        sim_p.iloc[:, i] = np.exp(sim.iloc[:, i]) * sim_p.iloc[:, i-1]

sim_10 = sim_p.iloc[:, -1]

ttm = []
for m in exp_date:
    if type(m) == str:
        m_split = m.split('/')
        delta = datetime(int(m_split[2]), int(m_split[0]), int(m_split[1])) - datetime(2023, 3, 3)
        ttm.append((delta.days + 10) / 365)
    else:
        ttm.append(0)

port_value_df = pd.DataFrame(0, index=port, columns=np.linspace(0, 10000, 1))

for i in range(len(sim_10.tolist())):
    values = portfolio_value(port, Type, Holding, option, sim_10[i], X, ivol_ls, ttm, r, b)
    port_value = pd.DataFrame(values, index=port)
    port_value_df[i] = port_value
