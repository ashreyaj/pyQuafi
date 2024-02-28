import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S0, K, T, r, sigma):
        """
        Args:
        S (float): price of underlying
        K (float): strike price of option
        T (float): time to maturity
        r (float): risk-free interest rate
        sigma (float): volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def callPrice(self):
        dplus = (1 / (self.sigma * np.sqrt(self.T))) * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2)*self.T)
        dminus = (1 / (self.sigma * np.sqrt(self.T))) * (np.log(self.S0 / self.K) + (self.r - self.sigma**2 / 2)*self.T)
        return self.S0 * norm.cdf(dplus) - self.K * np.exp(-self.r * self.T) * norm.cdf(dminus)
    
    def putPrice(self):
        dplus = (1 / (self.sigma * np.sqrt(self.T))) * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2)*self.T)
        dminus = (1 / (self.sigma * np.sqrt(self.T))) * (np.log(self.S0 / self.K) + (self.r - self.sigma**2 / 2)*self.T)
        return - self.S0 * norm.cdf(-dplus) + self.K * np.exp(-self.r * self.T) * norm.cdf(-dminus)
    
if __name__ == '__main__':
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    bs = BlackScholes(S0, K, T, r, sigma)
    call = bs.callPrice()
    put = bs.putPrice()
    print(f"Call price: {call}")
    print(f"Put price: {put}")