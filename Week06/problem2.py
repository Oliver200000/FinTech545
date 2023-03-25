import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Read the CSV file
options_df = pd.read_csv("AAPL_Options.csv")

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

# Function to calculate the implied volatility
def implied_volatility(price_function, market_price, stock_price, strike_price, risk_free_rate, dividend_yield, time_to_expiry):
    def objective_function(iv):
        return price_function(stock_price, strike_price, risk_free_rate, dividend_yield, iv, time_to_expiry) - market_price

    try:
        return brentq(objective_function, 1e-6, 2)
    except ValueError:
        return np.nan

# Current data and variables
stock_price = 151.03
current_date = datetime(2023, 3, 3)
risk_free_rate = 0.0425
dividend_yield = 0.0053

# Calculate the implied volatility for each option
options_df['Implied Volatility'] = options_df.apply(
    lambda row: implied_volatility(
        black_scholes_call if row['Type'] == 'Call' else black_scholes_put,
        row['Last Price'],
        stock_price,
        row['Strike'],
        risk_free_rate,
        dividend_yield,
        (datetime.strptime(row['Expiration'], '%m/%d/%Y') - current_date).days / 365
    ),
    axis=1
)

# Save the results to a new CSV file
options_df.to_csv("AAPL_Options_with_IV.csv", index=False)

# Print the DataFrame with implied volatilities
print(options_df)


# Separate call and put options
call_options = options_df[options_df['Type'] == 'Call']
put_options = options_df[options_df['Type'] == 'Put']


# Sort call and put options by strike price
call_options = call_options.sort_values('Strike')
put_options = put_options.sort_values('Strike')

# Plot implied volatility vs strike price
plt.plot(call_options['Strike'], call_options['Implied Volatility'], '-o', label='Call Options')
plt.plot(put_options['Strike'], put_options['Implied Volatility'], '-o', label='Put Options')

plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility vs Strike Price')
plt.legend()
plt.show()