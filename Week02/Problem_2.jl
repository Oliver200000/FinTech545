using CSV
using Distributions
using DataFrames
using StatsPlots

prob2 = CSV.read("/Users/oliver/Desktop/FinTech 545/Project/problem2.csv",DataFrame)

X = hcat(ones(size(prob2,1)),prob2.x)
Y = prob2.y

# Perform OLS estimation
b_hat = inv(X'*X)*X'*Y

println("OLS: ", b_hat)

error_vector = Y - X*b_hat

# create density plot
density(error_vector, xlabel = "Error", ylabel = "Density")
histogram(error_vector, bins = 20, xlabel = "Error", ylabel = "Frequency")


#MLE for Regression

Beta = b_hat
x = X
y = Y

function myll(s, b...)
    n = size(y,1)
    beta = collect(b)
    e = y - x*beta
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - e'*e/(2*s2)
    return ll
end

#MLE Optimization problem
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, beta[i=1:2],start=0)
    @variable(mle, σ >= 0.0, start = 1.0)

    register(mle,:ll,3,myll;autodiff=true)

    @NLobjective(
        mle,
        Max,
        ll(σ,beta...)
    )
##########################
optimize!(mle)

println("Betas: ", value.(beta))
