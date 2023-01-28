using Distributions
using HypothesisTests
using DataFrames
using Plots
using BenchmarkTools
using StatsBase


#Sample the kurtosis
kurtosis_samples = [moment(randn(100_000), 4)/(std(randn(100_000))^4) for __ in 1:100]

# Calculate the mean kurtosis 𝑘 and standard deviation 𝑆𝑘
k_mean = mean(kurtosis_samples)
k_std = std(kurtosis_samples)

#Calculate the T statistic (μ0 = 0)
t = (k_mean - 0) / (k_std / sqrt(100))

#Use the CDF function to find the p-value of the absolute value of the statistic and subtract from 1. Multiply the value by 2 because this is a 2 sided test.

p_value = 2 * (1 - cdf(TDist(99), abs(t)))

#If the value is lower than your threshold (typically 5%), then you reject the hypothesis that the kurtosis function is unbiased.
if p_value < 0.05
    println("Reject the null hypothesis that the kurtosis function is unbiased.")
else
    println("Fail to reject the null hypothesis that the kurtosis function is unbiased.")
end



#Sample the skewness
skewness_samples = [moment(randn(100_000), 3)/(std(randn(100_000))^3) for __ in 1:100]

#Calculate the mean skewness and standard deviation
mean_skewness = mean(skewness_samples)
std_skewness = std(skewness_samples)

#Calculate the T statistic (μ0 = 0)
T_statistic = (mean_skewness - 0) / (std_skewness / sqrt(100))

#Use the CDF function to find the p-value of the absolute value of the statistic and subtract from 1. Multiply the value by 2 because this is a 2-sided test.
p_value = 2 * (1 - cdf(TDist(99), abs(T_statistic)))

#If the value is lower than your threshold (typically 5%), then you reject the hypothesis that the skewness function is unbiased.

if p_value < 0.05
    println("Reject the null hypothesis that the skewness function is unbiased")
else
    println("Fail to reject the null hypothesis that the skewness function is unbiased")
end


