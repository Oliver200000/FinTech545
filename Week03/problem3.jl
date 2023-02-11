using DataFrames
using LinearAlgebra
using Statistics
using Plots
using CSV
using Random

# Read data from the file
data = CSV.read("/Users/oliver/Downloads/DailyReturn.csv", DataFrame)


# Define the function to calculate exponential weights
function populateWeights(dates, λ)
    n = size(dates, 1)
    w = zeros(n)
    cw = zeros(n)
    tw = 0
    
    # Calculate the weight for each stock, and the total weight and cumulative weight for each stock
    for i in 1:n
        individual_w = (1 - λ) * λ^(i - 1)
        w[i] = individual_w
        tw += individual_w
        cw[i] = tw
    end
    
    # Normalize the weights and cumulative weights for each stock
    for i in 1:n
        w[i] /= tw
        cw[i] /= tw
    end
    
    return w, cw
end


function exwCovMat(data, weights)
    stock_names = names(data)[2:end]
    n = size(stock_names, 1)
    w_cov_mat = zeros(n, n)
    
    # Calculate the variances and covariances
    for i in 1:n
        i_data = data[:, i + 1]
        i_mean = mean(i_data)
        
        for j in 1:n
            j_data = data[:, j + 1]
            j_mean = mean(j_data)
            
            sum = 0
            m = size(data, 1)
            for z in 1:m
                part = weights[z] * (i_data[z] - i_mean) * (j_data[z] - j_mean)
                sum += part
            end
            
            w_cov_mat[i, j] = sum
        end
    end
    
    return w_cov_mat
end

# Method 1: Standard Pearson correlation/variance
correlation_matrix = cor(Matrix(data[:, 2:end]))
variance_vector = var(Matrix(data[:, 2:end]))

# Method 2: Exponentially weighted λ = 0.97
λ = 0.97
dates = data[:, 1]
weights, cum_weights = populateWeights(dates, λ)
weights = reverse(weights)
w_correlation_matrix = exwCovMat(data, weights) ./ sqrt.(var(Matrix(data[:, 2:end]))' * var(Matrix(data[:, 2:end])))
w_variance_vector = diag(exwCovMat(data, weights))

# Combine the Pearson correlation matrix and the standard variance vector
covariance_matrix1 = correlation_matrix .* (sqrt.(variance_vector) * sqrt.(variance_vector)')

# Combine the Pearson correlation matrix and the EW variance vector
covariance_matrix2 = correlation_matrix .* (sqrt.(w_variance_vector) * sqrt.(w_variance_vector)')

# Combine the EW correlation matrix and the standard variance vector
covariance_matrix3 = w_correlation_matrix .* (sqrt.(variance_vector) * sqrt.(variance_vector)')

# Combine the EW correlation matrix and the EW variance vector
covariance_matrix4 = w_correlation_matrix .* (sqrt.(w_variance_vector) * sqrt.(w_variance_vector)')

# Define a function for direct simulation
function directSimulation(covariance_matrix, num_draws)
    num_stocks = size(covariance_matrix, 1)
    random_draws = randn(num_stocks, num_draws)
    return covariance_matrix * random_draws
end

# Simulate 25,000 draws from each covariance matrix using Direct Simulation
num_draws = 25000
direct_simulation1 = directSimulation(covariance_matrix1, num_draws)
direct_simulation2 = directSimulation(covariance_matrix2, num_draws)
direct_simulation3 = directSimulation(covariance_matrix3, num_draws)
direct_simulation4 = directSimulation(covariance_matrix4, num_draws)

plot(direct_simulation1, title="Direct Simulation - Covariance Matrix 1", legend=false, linealpha=0.5)
plot!(direct_simulation2, title="Direct Simulation - Covariance Matrix 2", legend=false, linealpha=0.5)
plot!(direct_simulation3, title="Direct Simulation - Covariance Matrix 3", legend=false, linealpha=0.5)
plot!(direct_simulation4, title="Direct Simulation - Covariance Matrix 4", legend=false, linealpha=0.5)