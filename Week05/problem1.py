import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV
data = pd.read_csv('/Users/oliver/Desktop/FinTech 545/Week05/Project/problem1.csv')




# Fit Normal Distribution
mu, std = stats.norm.fit(data)
norm_dist = stats.norm(mu, std)
sim_norm = np.random.normal(loc=mu, scale=std, size=10000)

# Fit Generalized T Distribution
df, mu, std = stats.t.fit(data)
gtd_dist = stats.t(df, mu, std)
sim_t = stats.t.rvs(df, loc=mu, scale=std, size=10000, random_state=None)
alpha = 0.05

# Calculate VaR and ES for Normal Distribution

var_norm = norm_dist.ppf(alpha)
es_norm = -sim_norm[sim_norm <= var_norm].mean().item()
var_norm = - var_norm 

# Calculate VaR and ES for Generalized T Distribution
var_gtd = gtd_dist.ppf(alpha)
es_gtd = -sim_t[sim_t <= var_gtd].mean().item()
var_gtd = -var_gtd 
# Print results
print("Normal Distribution:")
print(f"VaR: {var_norm:.4f}")
print(f"ES: {es_norm:.4f}\n")

print("Generalized T Distribution:")
print(f"VaR: {var_gtd:.4f}")
print(f"ES: {es_gtd:.4f}")



# plot distributions and ES and VaR
plt.figure()
# plot original data
sns.displot(data, stat='density', palette=('Greys'), label='Original Data')
# plot simulation
sns.kdeplot(sim_norm, color="b", label='Normal')
sns.kdeplot(sim_t, color="r", label='T')
# plot ES and VaR onto the graph
plt.axvline(x=-var_norm, color='b', label='var_norm')
plt.axvline(x=-var_gtd, color='r', label='var_gtd')
plt.axvline(x=-es_norm, color='b', label='es_norm', linestyle="dashed")
plt.axvline(x=-es_gtd, color='r', label='es_gtd', linestyle="dashed")
plt.legend()
plt.show()
