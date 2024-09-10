import numpy as np
from scipy.stats import norm

class Options:
    def __init__(self, S0, K, T, r):
        """
        Args:
        S (float): price of underlying
        K (float): strike price of option
        T (float): time to maturity
        r (float): risk-free interest rate
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r

class BlackScholes(Options):
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
        super().__init__(S0, K, T, r)
        self.t = t
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
    
class MonteCarlo(Options):
    def __init__(self, S0, K, T, t, r, drift, sigma, iterations):
        super().__init__(S0, K, T, r)
        self.t = t
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

class Binomial(Options):
    def __init__(self, S0, K, T, r, N, u, type="C"):
        super().__init__(S0, K, T, r)
        self.N = N
        self.u = u
        self.type = type
        self.d = 1/u
        self.dt = T/N
        self.p = (np.exp(r*self.dt) - self.d)/(self.u - self.d)
        self.dis = np.exp(-r*self.dt)

    def underlying_at_maturity(self):
        S = np.zeros(self.N+1)
        S[0] = S0 * self.d**self.N
        for i in range(1,self.N+1):
            S[i] = S[i-1] * self.u/self.d
        return S

    def option_at_maturity(self):
        S = self.underlying_at_maturity()
        C = np.zeros(self.N+1)
        for j in range(self.N+1):
            if self.type=="C":
                C[j] = max(S[j]-self.K, 0)
            elif self.type=="P":
                C[j] = max(self.K-S[j], 0)
            else:
                print("Error: Option type should be C or P")
                exit()
        return C

    def option_price(self):
        C = self.option_at_maturity()
        # calculate option price via back-propagation
        for i in np.arange(self.N,0,-1):
            for j in range(i):
                C[j] = self.dis * (self.p*C[j+1] + (1-self.p)*C[j])
        return C[0]

if __name__ == '__main__':
    S0 = 100
    K = 100
    T = 1
    t = 0.7
    r = 0.06
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

    N = 10
    u = 1.03
    opt_type = "C"
    bin = Binomial(S0, K, T, r, N, u, opt_type)
    call = bin.option_price()
    opt_type = "P"
    bin = Binomial(S0, K, T, r, N, u, opt_type)
    put = bin.option_price()
    print(f"Call price (Binomial): {call}")
    print(f"Put price (Binomial): {put}")

    