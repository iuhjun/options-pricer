import numpy as np
import matplotlib.pyplot as plt
from option import EuropeanOption

spot_range = np.linspace(50, 150, 100)

deltas = []
gammas = []

for S in spot_range:
    opt = EuropeanOption(S=S, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
    deltas.append(opt.delta())
    gammas.append(opt.gamma())

plt.figure()
plt.plot(spot_range, deltas)
plt.axvline(x=100, color='red', linestyle='--', label='Strike (K=100)')
plt.xlabel('Spot price (S)')
plt.ylabel('Delta')
plt.title('Delta vs Spot Price')
plt.legend()
plt.savefig('delta_plot.png')
plt.show()

plt.figure()
plt.plot(spot_range, gammas)
plt.axvline(x=100, color='red', linestyle='--', label='Strike (K=100)')
plt.xlabel('Spot price (S)')
plt.ylabel('Gamma')
plt.title('Gamma vs Spot Price')
plt.legend()
plt.savefig('gamma_plot.png')
plt.show()
