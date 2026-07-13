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
