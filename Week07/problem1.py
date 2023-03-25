# Import packages
from scipy.stats import norm
from datetime import datetime
import numpy as np
import inspect

# Constants and Variables
current_date = datetime(2022, 3, 13)
expiry_date = datetime(2022, 4, 15)
time_to_expiry = (expiry_date - current_date).days / 365
stock_price = 165
strike_price = 165
volatility = 0.2
interest_rate = 0.0425
dividend_yield = 0.0053
cost_of_carry = interest_rate - dividend_yield

# Function definitions
def compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry):
    return (np.log(stock_price / strike_price) + (cost_of_carry + volatility ** 2 / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))

def compute_d2(d1, time_to_expiry, volatility):
    return d1 - volatility * np.sqrt(time_to_expiry)

def gbsm_option_delta(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    is_call = 1 if option_type == "Call" else -1
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    delta = norm.cdf(d1 * is_call, 0, 1) * is_call
    return delta

def gbsm_option_gamma(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    gamma = norm.pdf(d1, 0, 1) / (stock_price * volatility * np.sqrt(time_to_expiry))
    return gamma

def gbsm_option_vega(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    vega = stock_price * norm.pdf(d1, 0, 1) * np.sqrt(time_to_expiry)
    return vega

def gbsm_option_theta(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    is_call = 1 if option_type == "Call" else -1
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    d2 = compute_d2(d1, time_to_expiry, volatility)
    theta = -stock_price * np.exp((cost_of_carry - interest_rate) * time_to_expiry) * norm.pdf(d1, 0, 1) * volatility / (2 * np.sqrt(time_to_expiry)) \
            -(cost_of_carry - interest_rate) * stock_price * np.exp((cost_of_carry - interest_rate) * time_to_expiry) * norm.cdf(d1 * is_call, 0, 1) * is_call \
            -interest_rate * strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2 * is_call, 0, 1) * is_call
    return theta

def gbsm_option_rho(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    is_call = 1 if option_type == "Call" else -1
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    d2 = compute_d2(d1, time_to_expiry, volatility)
    rho = strike_price * time_to_expiry * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2 * is_call, 0, 1) * is_call
    return rho

def gbsm_option_carry_rho(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    is_call = 1 if option_type == "Call" else -1
    d1 = compute_d1(stock_price, strike_price, time_to_expiry, volatility, cost_of_carry)
    carry_rho = stock_price * time_to_expiry * np.exp((cost_of_carry - interest_rate) * time_to_expiry) * norm.cdf(d1 * is_call, 0, 1) * is_call
    return carry_rho

# Calculate first-order derivative
def first_order_derivative(func, x, delta):
    return (func(x + delta) - func(x - delta)) / (2 * delta)

# Calculate second-order derivative
def second_order_derivative(func, x, delta):
    return (func(x + delta) + func(x - delta) - 2 * func(x)) / delta ** 2

def calculate_partial_derivative(func, order, variable_name, delta=1e-3):
    # Initialize argument names and order
    argument_names = list(inspect.signature(func).parameters.keys())
    derivative_functions = {1: first_order_derivative, 2: second_order_derivative}

    def partial_derivative(*args, **kwargs):
        # Parse argument names and order
        arguments_dict = dict(list(zip(argument_names, args)) + list(kwargs.items()))
        variable_value = arguments_dict.pop(variable_name)

        def partial_function(x):
            partial_kwargs = {variable_name: x, **arguments_dict}
            return func(**partial_kwargs)

        return derivative_functions[order](partial_function, variable_value, delta)

    return partial_derivative

def gbsm_option_price(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry):
    d1 = (np.log(stock_price / strike_price) + (cost_of_carry + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    is_call = 1 if option_type == "Call" else -1

    price = is_call * (stock_price * np.exp((cost_of_carry - interest_rate) * time_to_expiry) * norm.cdf(is_call * d1) - strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(is_call * d2))
    return price

# Delta
delta_call = gbsm_option_delta("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
delta_put = gbsm_option_delta("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_delta_numeric = calculate_partial_derivative(gbsm_option_price, 1, 'stock_price')
delta_call_numeric = gbsm_delta_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
delta_put_numeric = gbsm_delta_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(delta_call, delta_put)
print(delta_call_numeric, delta_put_numeric)

# Gamma
gamma_call = gbsm_option_gamma("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gamma_put = gbsm_option_gamma("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_gamma_numeric = calculate_partial_derivative(gbsm_option_price, 2, 'stock_price')
gamma_call_numeric = gbsm_gamma_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gamma_put_numeric = gbsm_gamma_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(gamma_call, gamma_put)
print(gamma_call_numeric, gamma_put_numeric)


# Vega
vega_call = gbsm_option_vega("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
vega_put = gbsm_option_vega("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_vega_numeric = calculate_partial_derivative(gbsm_option_price, 1, 'volatility')
vega_call_numeric = gbsm_vega_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
vega_put_numeric = gbsm_vega_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(vega_call, vega_put)
print(vega_call_numeric, vega_put_numeric)

# Theta
theta_call = gbsm_option_theta("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
theta_put = gbsm_option_theta("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_theta_numeric = calculate_partial_derivative(gbsm_option_price, 1, 'time_to_expiry')
theta_call_numeric = -gbsm_theta_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
theta_put_numeric = -gbsm_theta_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(theta_call, theta_put)
print(theta_call_numeric, theta_put_numeric)

# Rho
rho_call = gbsm_option_rho("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
rho_put = gbsm_option_rho("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_rho_numeric = calculate_partial_derivative(gbsm_option_price, 1, 'interest_rate')
rho_call_numeric = gbsm_rho_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
rho_put_numeric = gbsm_rho_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(rho_call, rho_put)
print(rho_call_numeric, rho_put_numeric)

# Carry Rho
carry_rho_call = gbsm_option_carry_rho("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
carry_rho_put = gbsm_option_carry_rho("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
gbsm_carry_rho_numeric = calculate_partial_derivative(gbsm_option_price, 1, 'cost_of_carry')
carry_rho_call_numeric = gbsm_carry_rho_numeric("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
carry_rho_put_numeric = gbsm_carry_rho_numeric("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, cost_of_carry)
print(carry_rho_call, carry_rho_put)
print(carry_rho_call_numeric, carry_rho_put_numeric)


def n_nodes(n_steps):
    return (n_steps + 2) * (n_steps + 1) // 2

def node_index(i, j):
    return n_nodes(j - 1) + i

def binomial_tree_no_div(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, n_steps):
    is_call = 1 if option_type == "Call" else -1
    dt = time_to_expiry / n_steps
    discount_factor = np.exp(-interest_rate * dt)
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(interest_rate * dt) - d) / (u - d)
    
    C = np.empty(n_nodes(n_steps), dtype=float)
            
    for i in np.arange(n_steps, -1, -1):
        for j in range(i, -1, -1):
            S = stock_price * u ** j * d ** (i - j)
            index = node_index(j, i)
            C[index] = max(0, (S - strike_price) * is_call)
            if i < n_steps:
                val = discount_factor * (p * C[node_index(j + 1, i + 1)] + (1 - p) * C[node_index(j, i + 1)])
                C[index] = max(C[index], val)
                
    return C[0]

def binomial_tree(option_type, stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, n_steps):
    if div_time is None or div is None:
        return binomial_tree_no_div(option_type, stock_price, strike_price, time_to_expiry, volatility, interest_rate, n_steps)
    
    is_call = 1 if option_type == "Call" else -1
    dt = time_to_expiry / n_steps
    discount_factor = np.exp(-interest_rate * dt)
    
    # Calculate u, d, and p
    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(interest_rate * dt) - d) / (u - d)

    new_time_to_expiry = time_to_expiry - div_time * dt
    new_n_steps = n_steps - div_time

    C = np.empty(n_nodes(div_time), dtype=float)
    for i in range(div_time, -1, -1):
        for j in range(i, -1, -1):
            S = stock_price * u ** j * d ** (i - j)
            val_exe = max(0, (S - strike_price) * is_call)
            if i < div_time:
                val = discount_factor * (p * C[node_index(j + 1, i + 1)] + (1 - p) * C[node_index(j, i + 1)])
            else:
                val = binomial_tree(option_type, S - div, strike_price, new_time_to_expiry, None, None, volatility, interest_rate, new_n_steps)
            C[node_index(j, i)] = max(val_exe, val)
    
    return C[0]

# Assume N is 200
N = 200
value_no_div_call = binomial_tree_no_div("Call", stock_price, strike_price, time_to_expiry, volatility, interest_rate, N)
value_no_div_put = binomial_tree_no_div("Put", stock_price, strike_price, time_to_expiry, volatility, interest_rate, N)
print("Binomial tree value without dividend for call: " + str(value_no_div_call))
print("Binomial tree value without dividend for put: " + str(value_no_div_put))

div_date = datetime(2022, 4, 11)
div = 0.88
div_time = int((div_date - current_date).days / (expiry_date - current_date).days * N)

value_call = binomial_tree("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
value_put = binomial_tree("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("Binomial tree value with dividend for call: " + str(value_call))
print("Binomial tree value with dividend for put: " + str(value_put))


# delta
cal_amr_delta_num = calculate_partial_derivative(binomial_tree, 1, 'stock_price')
delta_call_amr = cal_amr_delta_num("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
delta_put_amr = cal_amr_delta_num("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("American call delta with dividend: ", delta_call_amr)
print("American put delta with dividend: ", delta_put_amr)

# gamma
calculate_amr_gamma_num = calculate_partial_derivative(binomial_tree, 2, 'stock_price', delta=1)
gamma_call_amr = calculate_amr_gamma_num("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
gamma_put_amr = calculate_amr_gamma_num("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("American call gamma with dividend: ", gamma_call_amr)
print("American put gamma with dividend: ", gamma_put_amr)


# vega
calculate_amr_vega_num = calculate_partial_derivative(binomial_tree, 1, 'volatility')
vega_call_amr = calculate_amr_vega_num("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
vega_put_amr = calculate_amr_vega_num("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("American call vega with dividend: ", vega_call_amr)
print("American put vega with dividend: ", vega_put_amr)

# theta
calculate_amr_theta_num = calculate_partial_derivative(binomial_tree, 1, 'time_to_expiry')
theta_call_amr = -calculate_amr_theta_num("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
theta_put_amr = -calculate_amr_theta_num("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("American call theta with dividend: ", theta_call_amr)
print("American put theta with dividend: ", theta_put_amr)


# rho
calculate_amr_rho_num = calculate_partial_derivative(binomial_tree, 1, 'interest_rate')
rho_call_amr = calculate_amr_rho_num("Call", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
rho_put_amr = calculate_amr_rho_num("Put", stock_price, strike_price, time_to_expiry, div_time, div, volatility, interest_rate, N)
print("American call rho with dividend: ", rho_call_amr)
print("American put rho with dividend: ", rho_put_amr)


# sensitivity to change in dividend amount
# change the dividend amount on the first ex-dividend date by 1e-3
delta = 1e-3
call_value1 = binomial_tree("Call", stock_price, strike_price, time_to_expiry, div_time, div + delta, volatility, interest_rate, N)    
call_value2 = binomial_tree("Call", stock_price, strike_price, time_to_expiry, div_time, div - delta, volatility, interest_rate, N)    
call_sens_to_div_amount = (call_value1 - call_value2) / (2*delta)

put_value1 = binomial_tree("Put", stock_price, strike_price, time_to_expiry, div_time, div + delta, volatility, interest_rate, N)    
put_value2 = binomial_tree("Put", stock_price, strike_price, time_to_expiry, div_time, div - delta, volatility, interest_rate, N)    
put_sens_to_div_amount = (put_value1 - put_value2) / (2*delta)
print(f"Sensitivity to dividend amount: Call: {call_sens_to_div_amount:.3f}, Put: {put_sens_to_div_amount:.3f}")

