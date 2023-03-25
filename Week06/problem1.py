import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

stock_price = 165.0
strike_price = 165.0
risk_free_rate = 0.0425
dividend_yield = 0.0053

def time_to_maturity(start_date, end_date):
    return (end_date - start_date).days / 365


start_date = datetime(2023, 3, 3)
end_date = datetime(2023, 3, 17)
maturity = time_to_maturity(start_date, end_date)
print("Time to maturity: ", maturity)

days_to_expiry = (end_date - start_date).days
time_to_expiry = days_to_expiry / 365
iv_range = np.linspace(0.1, 0.8, 10)

def black_scholes_call(stock_price, strike_price, risk_free_rate, dividend_yield, iv, time_to_expiry):
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * iv ** 2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
    d2 = d1 - iv * np.sqrt(time_to_expiry)
    Nd1 = 0.5 * (1 + np.math.erf(d1 / np.sqrt(2)))
    Nd2 = 0.5 * (1 + np.math.erf(d2 / np.sqrt(2)))
    return stock_price * np.exp(-dividend_yield * time_to_expiry) * Nd1 - strike_price * np.exp(-risk_free_rate * time_to_expiry) * Nd2

def black_scholes_put(stock_price, strike_price, risk_free_rate, dividend_yield, iv, time_to_expiry):
    d1 = (np.log(stock_price / strike_price) + (risk_free_rate - dividend_yield + 0.5 * iv ** 2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
    d2 = d1 - iv * np.sqrt(time_to_expiry)
    Nd1 = 0.5 * (1 + np.math.erf(-d1 / np.sqrt(2)))
    Nd2 = 0.5 * (1 + np.math.erf(-d2 / np.sqrt(2)))
    return strike_price * np.exp(-risk_free_rate * time_to_expiry) * Nd2 - stock_price * np.exp(-dividend_yield * time_to_expiry) * Nd1

def plot_option_values(iv_range, call_values, put_values):
    plt.plot(iv_range, call_values, label='Call Option')
    plt.plot(iv_range, put_values, label='Put Option')
    plt.xlabel('Implied Volatility')
    plt.ylabel('Option Value')
    plt.legend()
    plt.show()

call_values = [black_scholes_call(stock_price, strike_price, risk_free_rate, dividend_yield, iv, time_to_expiry) for iv in iv_range]
put_values = [black_scholes_put(stock_price, strike_price, risk_free_rate, dividend_yield, iv, time_to_expiry) for iv in iv_range]
plot_option_values(iv_range, call_values, put_values)

call_values = [black_scholes_call(stock_price, strike_price + 20, risk_free_rate, dividend_yield, iv, time_to_expiry) for iv in iv_range]
put_values = [black_scholes_put(stock_price, strike_price - 20, risk_free_rate, dividend_yield, iv, time_to_expiry) for iv in iv_range]
plot_option_values(iv_range, call_values, put_values)