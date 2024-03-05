import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S0, K, T, t, r, sigma):
        """
        Args:
        S (float): price of underlying
        K (float): strike price of option
        T (float): time to maturity
        t (float): elapsed time
        r (float): risk-free interest rate
        sigma (float): volatility
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.t = t
        self.r = r
        self.sigma = sigma

    def call(self):
        dplus = (1 / (self.sigma * np.sqrt(self.T - self.t))) * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2)*(self.T - self.t))
        dminus = (1 / (self.sigma * np.sqrt(self.T - self.t))) * (np.log(self.S0 / self.K) + (self.r - self.sigma**2 / 2)*(self.T - self.t))
        return self.S0 * norm.cdf(dplus) - self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(dminus)
    
    def put(self):
        dplus = (1 / (self.sigma * np.sqrt(self.T - self.t))) * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2)*(self.T - self.t))
        dminus = (1 / (self.sigma * np.sqrt(self.T - self.t))) * (np.log(self.S0 / self.K) + (self.r - self.sigma**2 / 2)*(self.T - self.t))
        return - self.S0 * norm.cdf(-dplus) + self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(-dminus)
    
    def vega(self):
        dplus = (1 / (self.sigma * np.sqrt(self.T - self.t))) * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2)*(self.T - self.t))
        Nprime = (1/np.sqrt(2*np.pi)) * np.exp(-dplus**2/2)
        return self.S0 * Nprime * np.sqrt(self.T - self.t)
    
class MonteCarlo:
    def __init__(self, S0, K, T, t, r, drift, sigma, iterations):
        self.S0 = S0
        self.K = K
        self.T = T
        self.t = t
        self.r = r
        self.drift = drift
        self.sigma = sigma
        self.iterations = iterations

    def call(self):
        payoff = np.zeros((self.iterations, 2))

        fluctuations = np.random.normal(0, 1, [1, self.iterations])
        S = self.S0 * np.exp((self.drift - 0.5 * self.sigma**2) * (self.T - self.t) + self.sigma * np.sqrt(self.T - self.t) * fluctuations)

        payoff[:,1] = S - self.K
        payoffAvg = np.mean(np.max(payoff, axis=1))
        payoffAvg_pv = payoffAvg * np.exp(-self.r * (self.T - self.t)) # present value
        return payoffAvg_pv
    
    def put(self):
        payoff = np.zeros((self.iterations, 2))

        fluctuations = np.random.normal(0, 1, [1, self.iterations])
        S = self.S0 * np.exp((self.drift - 0.5 * self.sigma**2) * (self.T - self.t) + self.sigma * np.sqrt(self.T - self.t) * fluctuations)

        payoff[:,1] = self.K - S
        payoffAvg = np.mean(np.max(payoff, axis=1))
        payoffAvg_pv = payoffAvg * np.exp(-self.r * (self.T - self.t)) # present value
        return payoffAvg_pv
    
if __name__ == '__main__':
    S0 = 100
    K = 100
    T = 1
    t = 0.7
    r = 0.05
    sigma = 0.2
    bs = BlackScholes(S0, K, T, t, r, sigma)
    call = bs.call()
    put = bs.put()
    print(f"Call price (Black Scholes): {call}")
    print(f"Put price (Black Scholes): {put}")

    iterations = 10000
    mc = MonteCarlo(S0, K, T, t, r, r, sigma, iterations)
    call = mc.call()
    put = mc.put()
    print(f"Call price (Monte Carlo): {call}")
    print(f"Put price (Monte Carlo): {put}")
    