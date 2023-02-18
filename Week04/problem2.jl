using DataFrames
using CSV
using Statistics
using Distributions
using Optim
using LinearAlgebra
using TimeSeries

function calculate_returns(prices::DataFrame;
    returnMethod="DISCRETE",
    dateColumn="date")

    vars = names(prices)
    nVars = length(vars)
    vars = Symbol.(vars[vars .!= dateColumn])
    if nVars == length(vars)
        throw(ArgumentError("dateColumn: $dateColumn not in DataFrame: $vars"))
    end
    nVars = nVars - 1

    p = Matrix(prices[!, vars])
    n = size(p, 1)
    m = size(p, 2)
    p2 = similar(p)

    for j in 1:m
        p2[1, j] = p[1, j]
        for i in 2:n
            p2[i, j] = p[i, j] / p[i-1, j]
        end
    end

    if returnMethod == "DISCRETE"
        p2 = p2 .- 1.0
    elseif returnMethod == "LOG"
        p2 = log.(p2)
    else
        throw(ArgumentError("returnMethod must be DISCRETE or :log"))
    end

    dates = prices[2:n, dateColumn]
    out = DataFrame(dateColumn => dates)
    for i in 1:nVars
        out[!, vars[i]] = p2[2:end, i]
    end
    return out
end


df = CSV.read("/Users/oliver/Desktop/FinTech 545/FinTech-545-Spring2023-main/Week04/DailyPrices.csv",DataFrame)

returns = calculate_returns(df, returnMethod="DISCRETE",dateColumn="Date")

mean_META = mean(returns.META)
returns.META .-= mean_META

# Check if the mean of META is zero
if (mean(returns.META)<0.0000001)
    println("The mean of META is zero")
else
    println("The mean of META is not zero")
end


spy = returns.META
sd = std(spy)
VaR_05 = -quantile(Normal(0,sd),.05)

println(VaR_05)

function calculate_exponential_weights(data::Vector, lamb::Float64)
    lags = length(data)
    weights = Vector{Float64}(undef, lags)
    for i in 1:lags
        weight = (1 - lamb) * lamb^(i - 1)
        weights[i] = weight
    end
    weights = reverse(weights)
    normalized_weights = weights / sum(weights)
    return normalized_weights
end

function calculate_ewcov(data::Vector, lamb::Float64)
    weights = calculate_exponential_weights(data, lamb)
    error_matrix = data .- mean(data)
    ewcov = sum(weights .* error_matrix.^2)
    return ewcov
end

function calculate_var(data::Vector, mean::Real=0, alpha::Float64=0.05)
    return mean - quantile(data, alpha)
end

ew_cov = calculate_ewcov(returns.META, 0.94)
ew_variance = ew_cov[1, 1]
sigma = sqrt(ew_variance)
simulation_ew = randn(10_000) * sigma
var_ew = calculate_var(simulation_ew)
println(var_ew)




var_hist = calculate_var(returns.META)
print(var_hist)


