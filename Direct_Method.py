import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

'''
Monte Carlo Simulations WITHOUT Importance Sampling for Down-and-In Call Option Pricing
'''

def theoretical_price(S_0, K, B, r, sigma, T):
    d_b = (np.log(B**2/ (S_0*K)) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    alpha = (2*r) / sigma**2 + 1
    
    theoretical_price =  S_0 * (B/S_0)**alpha * norm.cdf(d_b) - K * np.exp(-r*T) * (B/S_0)**(alpha - 2) * norm.cdf(d_b - sigma * np.sqrt(T))

    return theoretical_price

def simulated_price(S_0, K, B, r, sigma, T, m, n_simulations):
    delta_t = T / m  # Time step
    b = np.log(S_0/B)  # Barrier factor, B = S_0 * exp(-b)
    c = np.log(K/S_0)  # Strike price factor, K = S_0 * exp(c)

    # Initialize normal random variables with
    X_n = np.random.normal(loc=(r - 0.5 * sigma**2) * delta_t, scale=np.sqrt(sigma**2 * delta_t), size=(n_simulations, m))
    U_n = np.cumsum(X_n, axis=1)
    barrier_hit = np.any(U_n <= -b, axis=1)  # Check barrier condition for all simulations
    final_prices = S_0 * np.exp(U_n[:, -1])
    payoffs = np.where(barrier_hit, np.maximum(final_prices - K, 0), 0)

    # Count number of non-zero payoffs
    num_nonzero_payoffs = np.count_nonzero(payoffs)
    rate_of_non_zero_payoffs = num_nonzero_payoffs / n_simulations

    # Compute discounted average payoff
    estimated_prices = np.exp(-r * T) * np.cumsum(payoffs) / np.arange(1, n_simulations + 1)

    return estimated_prices ,rate_of_non_zero_payoffs

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
b = np.log(S_0/B)  # Barrier factor, B = S_0 * exp(-b)
c = np.log(K/S_0)  # Strike price factor, K = S_0 * exp(c)
n_simulations = 100_000  # Total number of Monte Carlo simulations

theoretical_price = theoretical_price(S_0, K, B, r, sigma, T)
estimated_prices, rate_of_non_zero_payoffs = simulated_price(S_0, K, B, r, sigma, T, m, n_simulations)
simulated_price = estimated_prices[-1]  # Final estimate from the last simulation

# Plot convergence

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_simulations + 1), estimated_prices, label='Estimated Option Price', color='blue')
plt.axhline(y=simulated_price, color='red', linestyle='--', label='Final Estimate')
plt.axhline(y=theoretical_price, color='green', linestyle='-.', label='Theoretical Price')
plt.xlabel("Number of Simulations")
plt.ylabel("Option Price")
plt.title("Convergence of Down-and-In Call Option Price, B = 75, K = 96")
plt.legend()
plt.show()

print(f"Final Estimated Down-and-In Call Option Price: {simulated_price}")
print(f"Theoretical Down-and-In Call Option Price: {theoretical_price}")
print(f"Rate of Non-Zero Payoffs: {rate_of_non_zero_payoffs}")
print(f"Variance: {np.var(estimated_prices)}")