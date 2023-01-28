using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using PlotThemes
using Printf
using JuMP
using Ipopt
using StateSpaceModels
using MultivariateStats
using StatsPlots

#AR1
#y_t = 1.0 + 0.5*y_t-1 + e, e ~ N(0,0.1)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = 1.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 1:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*yt_last + e[i]
    yt_last = y_t
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))
println(@sprintf("Expected values Y: %.2f, %.4f",2.0,.01/(1-.5^2)))

plot(y,imgName="ar1_acf_pacf.png",title="AR 1")
ar1 = SARIMA(y,order=(1,0,0),include_mean=true)


StateSpaceModels.fit!(ar1)
print_results(ar1)



#AR2
#y_t = 1.0 + 0.5y_t-1 + 0.3y_t-2 + e
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last1 = 1.0
yt_last2 = 2.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 1:(n+burn_in)
    global yt_last1, yt_last2
    y_t = 1.0 + 0.5*yt_last1 + 0.3*yt_last2 + e[i]
    yt_last2 = yt_last1
    yt_last1 = y_t
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))

plot(y,imgName="ar2_acf_pacf.png",title="AR 2")

ar2 = SARIMA(y,order=(2,0,0),include_mean=true)

StateSpaceModels.fit!(ar2)
print_results(ar2)

#AR3
#yt = c + phi1 * yt-1 + phi2 * yt-2 +phi3 * yt-3 + e

n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last1 = 1.0
yt_last2 = 2.0
yt_last3 = 3.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)
c = 1
phi1 = 0.5
phi2 = 0.3
phi3 = 0.2

for i in 1:(n+burn_in)
    global yt_last1, yt_last2, yt_last3
    y_t = c + phi1*yt_last1 + phi2*yt_last2 + phi3*yt_last3 + e[i]
    yt_last3 = yt_last2
    yt_last2 = yt_last1
    yt_last1 = y_t
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))

plot(y,imgName="ar3_acf_pacf.png",title="AR 3")


#MA1
#y_t = 1.0 + .05*e_t-1 + e, e ~ N(0,.01)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = 1.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 2:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*e[i-1] + e[i]
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))
println(@sprintf("Expected values Y: %.2f, %.4f",1.0,(1+.5^2)*.01))

plot(y,imgName="ma1_acf_pacf.png",title="MA 1")

ma1 = SARIMA(y,order=(0,0,1),include_mean=true)

StateSpaceModels.fit!(ma1)
print_results(ma1)

#MA(2)
#yt = mu + e_t - theta1 * e_t-1 - theta2 * e_t-2
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

theta1 = 0.5
theta2 = -0.3
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 3:(n+burn_in)
    e_t = rand(d)
    y_t = e_t + theta1*e[i-1] + theta2*e[i-2]
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

# Plot PACF
plot(y)



#MA(3)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

theta1 = 0.5
theta2 = -0.3
theta3 = 0.2
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 4:(n+burn_in)
    e_t = rand(d)
    y_t = e_t + theta1*e[i-1] + theta2*e[i-2] + theta3*e[i-3]
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

plot(y)



