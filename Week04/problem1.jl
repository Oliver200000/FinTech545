using Statistics
using StatsPlots
using Distributions

sigma = 0.5
price_prev = 50

# return
r = rand(Normal(0, sigma), 10_000)

price_brownian = price_prev .+ r
price_arithmetic = price_prev .* (1 .+ r)
price_log = price_prev .* exp.(r)

println("For Classical Brownian Motion, the mean value is ",mean(price_brownian), ", the standard deviation is ", std(price_brownian))
println("For Arithmetic Return System, the mean value is ",mean(price_arithmetic), ", the standard deviation is ", std(price_arithmetic))
println("For Log Return or Geometric Brownian Motion ",mean(log.(price_log)), ", the standard deviation is ", std(log.(price_log)))


gr(size=(600,800))

p1 = plot(density(price_brownian), title="Classical Brownian Motion", xlabel="Price", ylabel="Density")
p2 = plot(density(price_arithmetic), title="Arithmetic Return System", xlabel="Price", ylabel="Density")
p3 = plot(density(price_log), title="Geometric Brownian Motion", xlabel="Price", ylabel="Density")

plot(p1, p2, p3, layout=(3,1), legend=false)