import matplotlib.pyplot as plt
from option import EuropeanOption

opt = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
bs_price = opt.black_scholes_price()

simulation_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
mc_prices = [opt.monte_carlo_price(n) for n in simulation_counts]

plt.plot(simulation_counts, mc_prices, marker='o', label='Monte Carlo price')
plt.axhline(y=bs_price, color='red', linestyle='--', label='Black-Scholes price')
plt.xscale('log')
plt.xlabel('Number of simulations')
plt.ylabel('Option price')
plt.title('Monte Carlo Convergence to Black-Scholes Price')
plt.legend()
plt.savefig('convergence_plot.png')
plt.show()