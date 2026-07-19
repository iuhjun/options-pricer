import numpy as np
from scipy.stats import norm

class EuropeanOption:
    """
    Represents a European option (call or put) that is priced under 
    Black-Scholes assumptions.
    """

    def __init__(self, S, K, T, r, sigma, option_type='call'):
        """
        Parameters:
            S (float): current spot price of the underlying asset
            K (float): strike price
            T (float): time to maturity (years)
            r (float): risk-free interest rate (annualised)
            sigma (float): volatility of the underlying asset (annualised)
            option_type (str): 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        if option_type not in ('call', 'put'):
            raise ValueError("option_type must be 'call' or 'put'")
        self.option_type = option_type

    def __repr__(self):
        return (f"EuropeanOption(S={self.S}, K={self.K}, T={self.T}, "
                f"r={self.r}, sigma={self.sigma}, type='{self.option_type}')")

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + ((self.sigma)**2) / 2) * self.T) / ((self.sigma) * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def black_scholes_price(self):
        d1 = self.d1()
        d2 = self.d2()

        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return price

    def simulate_terminal_prices(self, num_simulations):
        Z = np.random.standard_normal(num_simulations)
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        return S_T

    def monte_carlo_price(self, num_simulations):
        S_T = self.simulate_terminal_prices(num_simulations)

        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        discounted_price = np.exp(-self.r * self.T) * payoffs.mean()
        return discounted_price

    def delta(self):
        d1 = self.d1()
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self):
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)

# test
opt_call = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
print("Delta:", opt_call.delta())
print("Gamma:", opt_call.gamma())
print("Vega:", opt_call.vega())

opt_put = EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='put')
print("Delta:", opt_put.delta())