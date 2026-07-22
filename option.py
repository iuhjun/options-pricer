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
        """
        Compute the d1 term used in the Black-Scholes formula and Greeks.

        Returns:
            float: the d1 value for this option's parameters.
        """
        return (np.log(self.S / self.K) + (self.r + ((self.sigma)**2) / 2) * self.T) / ((self.sigma) * np.sqrt(self.T))

    def d2(self):
        """
        Compute the d2 term used in the Black-Scholes formula and Greeks.

        Returns:
            float: the d2 value for this option's parameters.
        """
        return self.d1() - self.sigma * np.sqrt(self.T)

    def black_scholes_price(self):
        """
        Compute the option price using the closed-form Black-Scholes formula.

        Returns:
            float: the theoretical price of the option.
        """
        d1 = self.d1()
        d2 = self.d2()

        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        
        return price

    def simulate_terminal_prices(self, num_simulations):
        """
        Simulate terminal stock prices at maturity using geometric Brownian motion under the risk-neutral measure.

        Parameters:
            num_simulations (int): number of simulated price paths to use.

        Returns:
            np.ndarray: array of simulated terminal stock prices, length num_simulations.
        """
        Z = np.random.standard_normal(num_simulations)
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        return S_T

    def monte_carlo_price(self, num_simulations):
        """
        Estimate the option price via Monte Carlo simulation: simulate terminal stock prices, compute the discounted 
        average payoff.

        Parameters:
            num_simulations (int): number of simulated price paths to use.

        Returns:
            float: the Monte Carlo estimate of the option price.
        """
        S_T = self.simulate_terminal_prices(num_simulations)

        if self.option_type == 'call':
            payoffs = np.maximum(S_T - self.K, 0)
        else:
            payoffs = np.maximum(self.K - S_T, 0)
        
        discounted_price = np.exp(-self.r * self.T) * payoffs.mean()
        return discounted_price

    def delta(self):
        """
        Compute delta (sensitivity of price to a change in spot price) using the closed-form Black-Scholes formula.

        Returns:
            float: delta, between 0 and 1 for a call, -1 and 0 for a put.
        """
        d1 = self.d1()
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self):
        """
        Compute gamma (sensitivity of delta to a change in spot price) using the closed-form Black-Scholes formula.

        Returns:
            float: gamma, always positive for calls and puts.
        """
        d1 = self.d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """
        Compute vega (sensitivity of price to a change in volatility) using the closed-form Black-Scholes formula.

        Returns:
            float: vega, always positive for calls and puts.
        """
        d1 = self.d1()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)
    
    def delta_fd(self, h=0.01):
        """
        Estimate delta numerically via central finite differences, by bumping the spot price up and down by h and 
        re-pricing.

        Parameters:
            h (float): size of the bump applied to the spot price.

        Returns:
            float: finite-difference estimate of delta.
        """
        option_up = EuropeanOption(self.S + h, self.K, self.T, self.r, self.sigma, self.option_type)
        option_down = EuropeanOption(self.S - h, self.K, self.T, self.r, self.sigma, self.option_type)
        return (option_up.black_scholes_price() - option_down.black_scholes_price()) / (2 * h)

    def gamma_fd(self, h=0.01):
        """
        Estimate gamma numerically via central finite differences, using the price at S+h, S, and S-h.

        Parameters:
            h (float): size of the bump applied to the spot price.

        Returns:
            float: finite-difference estimate of gamma.
        """
        option_up = EuropeanOption(self.S + h, self.K, self.T, self.r, self.sigma, self.option_type)
        option_down = EuropeanOption(self.S - h, self.K, self.T, self.r, self.sigma, self.option_type)
        return (option_up.black_scholes_price() - 2 * self.black_scholes_price() + option_down.black_scholes_price()) / (h ** 2)

    def vega_fd(self, h=0.01):
        """
        Estimate vega numerically via central finite differences, by bumping volatility up and down by h and re-pricing.

        Parameters:
            h (float): size of the bump applied to volatility.

        Returns:
            float: finite-difference estimate of vega.
        """
        option_up = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma + h, self.option_type)
        option_down = EuropeanOption(self.S, self.K, self.T, self.r, self.sigma - h, self.option_type)
        return (option_up.black_scholes_price() - option_down.black_scholes_price()) / (2 * h)
