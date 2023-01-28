import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

'''
# Simulate an AR(1) process
np.random.seed(123)
ar_params = [1, 0.8] # set the AR(1) parameters
ma_params = [1] # set the MA(0) parameters
ar = sm.tsa.ArmaProcess(ar_params, ma_params)
y = ar.generate_sample(nsample=100)

# Plot the ACF and PACF of the AR(1) process
_ = plot_acf(y, lags=20)
plt.show()
_ = plot_pacf(y, lags=20)
plt.show()
'''

'''
np.random.seed(1)

# Simulate AR(2) process
n = 1000
burn_in = 50

phi_1, phi_2 = 0.5, 0.3
sigma = 0.1

y = np.zeros(n)
e = np.random.normal(0, sigma, n+burn_in)

# Generate AR(2) process
for i in range(2, n+burn_in):
    y[i-burn_in] = 1 + phi_1*y[i-burn_in-1] + phi_2*y[i-burn_in-2] + e[i]

# Plot ACF and PACF
plot_acf(y)
plt.show()
plot_pacf(y, lags = 20)
plt.show()
'''

'''

np.random.seed(1)

# Simulate AR(3) process
n = 1000
burn_in = 50

phi_1, phi_2, phi_3 = 0.5, 0.3, 0.2
sigma = 0.1

y = np.zeros(n)
e = np.random.normal(0, sigma, n+burn_in)

# Generate AR(3) process
for i in range(3, n+burn_in):
    y[i-burn_in] = 1 + phi_1*y[i-burn_in-1] + phi_2*y[i-burn_in-2] + phi_3*y[i-burn_in-3] + e[i]

# Plot ACF and PACF
plot_acf(y, lags = 20)
plt.show()
plot_pacf(y, lags = 20)
plt.show()
'''

'''
np.random.seed(1)

# Simulate MA(1) process
n = 1000
burn_in = 50

theta = 0.5
sigma = 0.1

y = np.zeros(n)
e = np.random.normal(0, sigma, n+burn_in)

# Generate MA(1) process
for i in range(1, n+burn_in):
    y[i-burn_in] = e[i] + theta*e[i-1]

# Plot ACF and PACF
plot_acf(y, lags = 20)
plt.show()
plot_pacf(y, lags = 20)
plt.show()

'''

'''
np.random.seed(1)

# Simulate MA(2) process
n = 1000
burn_in = 50

theta_1, theta_2 = 0.5, 0.3
sigma = 0.1

e = np.random.normal(0, sigma, n+burn_in)
y = np.zeros(n)

# Generate MA(2) process
for i in range(2, n+burn_in):
    y[i-burn_in] = e[i] + theta_1*e[i-1] + theta_2*e[i-2]

# Plot ACF and PACF
plot_acf(y, lags = 20)
plt.show()
plot_pacf(y, lags = 20)
plt.show()
'''
np.random.seed(1)

# Simulate MA(3) process
n = 1000
burn_in = 50

theta_1, theta_2, theta_3 = 0.5, 0.3, 0.2
sigma = 0.1

e = np.random.normal(0, sigma, n+burn_in)
y = np.zeros(n)

# Generate MA(3) process
for i in range(3, n+burn_in):
    y[i-burn_in] = e[i] + theta_1*e[i-1] + theta_2*e[i-2] + theta_3*e[i-3]

# Plot ACF and PACF
plot_acf(y, lags = 20)
plt.show()
plot_pacf(y, lags = 20)
plt.show()