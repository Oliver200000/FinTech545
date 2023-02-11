using DataFrames
using LinearAlgebra
using Statistics
using Plots
using CSV


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

# Define the function to calculate the exponentially weighted covariance matrix
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


# Calculate the weights and cumulative weights
λ = 0.95
dates = data[:, 1]
weights, cum_weights = populateWeights(dates, λ)
weights = reverse(weights)

# Calculate the covariance matrix
covariance_matrix = exwCovMat(data, weights)


# Calculate the eigenvalues and eigenvectors of the covariance matrix, and sort the eigenvalues in descending order
e_val, e_vec = eigen(covariance_matrix)
sorted_e_val = sort(real(e_val), rev=true)

# Set negative eigenvalues to zero
sorted_e_val_real = real(sorted_e_val)
sorted_e_val_real[sorted_e_val_real .< 0] .= 0

# Calculate the percent of variance explained by each eigenvalue
e_sum = sum(sorted_e_val)
sub_sorted_e_val = sorted_e_val[sorted_e_val .> 1e-8]
individual_percent = sub_sorted_e_val ./ e_sum
total_percent = cumsum(individual_percent)

x = 1:length(total_percent)
y = total_percent
plot(x, y, marker = :circle)
xlabel!("count")
ylabel!("total_percent")
title!("λ = 0.95")