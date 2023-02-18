using CSV
using DataFrames


function pd_return_calculate(prices::Vector, method::String="arithmetic")
    price_change_percent = diff(prices) ./ prices[1:end-1]
    if method == "arithmetic"
        return price_change_percent .- 1
    elseif method == "log"
        return log.(price_change_percent)
    end
end

# Load data and calculate returns
prices = CSV.File("/Users/oliver/Desktop/FinTech 545/FinTech-545-Spring2023-main/Week04/DailyPrices.csv"; dateformat="yyy-mm-dd") |> DataFrame
portfolios = CSV.File("/Users/oliver/Desktop/FinTech 545/FinTech-545-Spring2023-main/Week04/Project/portfolio.csv") |> DataFrame
println(size(prices))
println(size(portfolios))
returns = pd_return_calculate(prices[:, 2], "log")

# Combine the portfolios to get a total one and append it to the end for easier calculation
total_holdings = combine(groupby(portfolios, :Stock), :Holding => sum => :Holding)
total_holdings[!, :Portfolio] .= "Total"
append!(portfolios, total_holdings)


function shapiro_test(data; alpha=0.05)
    test_stat, p = shapiro(data)
    if p > alpha
        return 1
    else
        return 0
    end
end

# Load data and calculate returns
prices = CSV.File("DailyPrices.csv"; dateformat="yyy-mm-dd") |> DataFrame
portfolios = CSV.File("portfolio.csv") |> DataFrame
returns = pd_return_calculate(prices[:, 2], "log")

# Determine if the returns are normally distributed using Shapiro-Wilks test
for (portfolio_index, portfolio) in groupby(portfolios, :Portfolio)
    portfolio_returns = returns[:, portfolio.Stock]
    num_normal = apply(portfolio_returns, 2, shapiro_test) |> sum
    percentage_normal = num_normal / size(portfolio_returns, 2) * 100
    println(string(percentage_normal) * "%")
end

current_prices = DataFrame(Price=prices[end, :])

for (portfolio_index, portfolio) in groupby(portfolios, :Portfolio)
    portfolio = portfolio |> setindex!(:Stock)
    portfolio = join(portfolio, current_prices[portfolio[!, :Stock], :])

    current_values = portfolio[!, :Holding] .* portfolio[!, :Price]
    portfolio_value = sum(current_values)

    sim_returns = returns[!, portfolio[!, :Stock]]
    sim_prices = (1 .+ sim_returns) .* portfolio[!, :Price]'
    sim_values = sim_prices * portfolio[!, :Holding]

    historic_var = calculate_var(sim_values, portfolio_value)
    println("Portfolio $(portfolio_index): $(historic_var)")
end