import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
Monte Carlo Simulations with Importance Sampling for Down-and-In Call Option Pricing
'''

# Parameters
# The Setting
r = 0.05  # Risk-free rate

# The Underlying Asset
S_0 = 95  # Initial asset price
sigma = 0.15  # Volatility

# The Option (Down-and-In Call Option)
B = 75 # Barrier level
K = 96 # Strike price
T = 1  # Time to maturity

# The Simulations
m = 1_000  # Number of time steps
delta_t = T / m  # Time step
b = np.log(S_0/B)  # Barrier factor, B = S_0 * exp(-b)
c = np.log(K/S_0)  # Strike price factor, K = S_0 * exp(c)
mu = (2*b + c)/ T # New mean of the normal random variables (Importance Sampling)
n_simulations = 10_000  # Total number of Monte Carlo simulations

# Importance Sampling (For Likelihood Ratio)
theta = (mu - r + 0.5 * sigma**2) / sigma**2 # Auxiliary variable
psi = (r - 0.5 * sigma**2) * delta_t * theta + 0.5 * sigma**2 * delta_t * theta**2 # Auxiliary variable

# Initialize normal random variables with
# mean -mu * delta_t (Negative Drift)
X_down = np.random.normal(loc= -mu * delta_t, scale=np.sqrt(sigma**2 * delta_t), size=(n_simulations, m))
# mean -mu * delta_t (Positivie Drift)
X_up = np.random.normal(loc= mu * delta_t, scale=np.sqrt(sigma**2 * delta_t), size=(n_simulations, m))

# Compute Cumulative Change
U_temp = np.cumsum(X_down, axis=1)

# Find the first index where the barrier is hit
hit_mask = U_temp <= -b
tau = np.where(hit_mask.any(axis=1), hit_mask.argmax(axis=1), -1) 
# Note that by letting tau = -1, we will also update values for simulations where the barrier is never hit 

# Create a mask for elements after hitting the barrier
row_indices = np.arange(X_down.shape[0])[:, None]
col_indices = np.arange(X_down.shape[1])
modification_mask = col_indices > tau[:, None]  # True where we should modify
# In "modification_mask", if we have all "True", it means the barrier was never hit

# Apply modifications
X_n = np.where(modification_mask, X_up, X_down)
U_n = np.cumsum(X_n, axis=1)
final_prices = S_0 * np.exp(U_n[:, -1])
payoffs = np.where(tau >= 0, np.maximum(final_prices - K, 0), 0)

# Likelihood Ratio
tau_temp = np.where(tau >= 0, tau, 0)
tau_mask = col_indices == tau_temp[:, None]
U_tau = U_n[tau_mask]
L = np.exp(2*mu/sigma**2 * U_tau - theta * U_n[:, -1] + m * psi) # Likelihood Ratio

# Count number of non-zero payoffs
num_nonzero_payoffs = np.count_nonzero(payoffs)

# Compute discounted average payoff
adjusted_payoffs = L * payoffs
average_adjusted_payoffs = np.mean(L * payoffs)
option_price = np.exp(-r * T) * average_adjusted_payoffs

print(option_price)

# Calculation of the Theoretical Price
alpha = 2*r/sigma**2 + 1
d_b = ((-c-2*b) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
theoretical_price = S_0 * np.exp(-b * alpha) * norm.cdf(d_b) - S_0 * np.exp(c-r*T-b * (alpha - 2)) * norm.cdf(d_b - sigma * np.sqrt(T))

# Plot convergence
estimated_prices = np.cumsum(adjusted_payoffs) / np.arange(1, n_simulations + 1)
discounted_prices = np.exp(-r * T) * estimated_prices

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_simulations + 1), discounted_prices, label='Estimated Option Price', color='blue')
plt.axhline(y=option_price, color='red', linestyle='--', label='Final Estimate')
plt.axhline(y=theoretical_price, color='green', linestyle='-.', label='Theoretical Price')
plt.xlabel("Number of Simulations")
plt.ylabel("Option Price")
plt.title("Convergence of Down-and-In Call Option Price, B = 75, K = 96 with Importance Sampling")
plt.legend()
plt.show()

print(f"Final Estimated Down-and-In Call Option Price: {option_price}")
print(f"Theoretical Down-and-In Call Option Price: {theoretical_price}")
print(f"Rate of Non-Zero Payoffs: {num_nonzero_payoffs/n_simulations}")
print(f"Variance: {np.var(discounted_prices)}")