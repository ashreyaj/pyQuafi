import numpy as np
from scipy.stats import norm
from scipy.special import laguerre
import statsmodels.api as sm
from wiener import GeometricBrownianMotion

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
    def __init__(self, S0, K, T, r, N, type="C", parameter_model=None, sigma=None, u=None):
        super().__init__(S0, K, T, r)
        self.N = N
        self.type = type
        self.parameter_model = parameter_model
        self.dt = T/N
        self.dis = np.exp(-r*self.dt)
        self.sigma = sigma
        self.d = None
        
        if parameter_model is None:
            if u is None:
                print("For generic binomial tree u value must be given")
                exit()
            self.u = u
            self.d = 1/u
            self.p = (np.exp(r*self.dt) - self.d)/(self.u - self.d)
        elif self.parameter_model=="CRR":
            self.crr()
        elif self.parameter_model=="JR":
            self.jr()
        elif self.parameter_model=="EQP":
            self.eqp()
        elif self.parameter_model=="TRG":
            self.trg()

    def crr(self):
        if self.sigma is None:
            print("Set value for sigma")
            exit()
        self.u = np.exp(self.sigma*np.sqrt(self.dt))
        self.d = 1/self.u
        self.p = (np.exp(r*self.dt) - self.d)/(self.u - self.d)

    def jr(self):
        if self.sigma is None:
            print("Set value for sigma")
            exit()
        nu = self.r - 0.5*self.sigma**2
        self.u = np.exp(nu*self.dt + self.sigma*np.sqrt(self.dt))
        self.d = np.exp(nu*self.dt - self.sigma*np.sqrt(self.dt))
        self.p = 0.5

    def eqp(self):
        if self.sigma is None:
            print("Set value for sigma")
            exit()
        nu = self.r - 0.5*self.sigma**2
        dxu = 0.5*nu*self.dt + 0.5*np.sqrt(4*self.sigma**2*self.dt - 3*(nu*self.dt)**2)
        dxd = 1.5*nu*self.dt - 0.5*np.sqrt(4*self.sigma**2*self.dt - 3*(nu*self.dt)**2)
        self.p = 0.5
        return dxu, dxd
    
    def trg(self):
        nu = self.r - 0.5*self.sigma**2
        dxu = np.sqrt(sigma**2*self.dt + nu**2*self.dt**2)
        dxd = -dxu
        self.p = 0.5 + 0.5*nu*self.dt/dxu
        return dxu, dxd

    def underlying_at_maturity(self):
        S = np.zeros(self.N+1)

        if self.parameter_model == "EQP" or self.parameter_model == "TRG":
            dxu, dxd = self.eqp()
            S[0] = S0*np.exp(N*dxd)
            for j in range(1,N+1):
                S[j] = S[j-1]*np.exp(dxu - dxd)

        else:
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
    
class LongstaffSchwartz(MonteCarlo):
    def __init__(self, S0, K, T, t, r, drift, sigma, iterations, steps, type="C", basis_fn="Laguerre", order=5):
        super().__init__(S0, K, T, t, r, drift, sigma, iterations)
        self.steps = steps
        self.type = type
        self.basis_fn = basis_fn
        self.order = order
        self.basis()
        self.paths()
        self.payoff()

    def basis(self):
        if self.basis_fn == "Laguerre":
            self.basis = [laguerre(i) for i in range(self.order)]
        else:
            raise Exception("Basis function not implemented.")
        
    def paths(self):
        dt = T / self.steps
        paths = np.zeros((self.iterations, self.steps))
        for i in range(self.iterations):
            gbm = GeometricBrownianMotion(dt, self.S0, self.drift, self.sigma, self.steps)
            time, price = gbm.gbm()
            paths[i] = price
        self.dt = dt
        self.time = time
        self.paths = paths
    
    def payoff(self):
        if self.type == "C":
            payoff = np.maximum(self.paths - self.K, 0)
        else:
            payoff = np.maximum(self.K - self.paths, 0)
        self.payoff = payoff
    
    def option_price(self):
        value = self.payoff[:, -1]
        for i in range(self.steps-1, 0, -1):
            itm = self.payoff[:, i] > 0
            if not np.any(itm):
                continue
            X = np.column_stack(([b(self.paths[itm, i]) for b in self.basis]))
            Y = self.payoff[itm, i]
            model = sm.OLS(Y, X).fit()
            continuation = model.predict(X)
            exercise = self.payoff[itm, i] > continuation
            value[itm] = np.where(exercise, self.payoff[itm, i], value[itm]*np.exp(-self.r*self.dt))

        return np.mean(value) * np.exp(-self.r*self.dt)

if __name__ == '__main__':
    S0 = 100
    K = 110
    T = 1
    t = 0.0
    r = 0.06
    sigma = 0.3

    # # --- Black Scholes model ---
    bs = BlackScholes(S0, K, T, t, r, sigma)
    call = bs.call()
    put = bs.put()
    print(f"Call price (Black Scholes): {call}")
    print(f"Put price (Black Scholes): {put}\n")

    # --- Monte Carlo method ---
    iterations = 10000
    mc = MonteCarlo(S0, K, T, t, r, r, sigma, iterations)
    call = mc.call()
    put = mc.put()
    print(f"Call price (Monte Carlo): {call}")
    print(f"Put price (Monte Carlo): {put}\n")

    # --- Binomial tree ---
    T = 1
    N = 1000
    opt_type = "C"
    parameter_model = "TRG"
    bin = Binomial(S0, K, T, r, N, opt_type, parameter_model, sigma)
    call = bin.option_price()
    opt_type = "P"
    bin = Binomial(S0, K, T, r, N, opt_type, parameter_model, sigma)
    put = bin.option_price()
    print(f"Call price (Binomial - {parameter_model} method): {call}")
    print(f"Put price (Binomial - {parameter_model} method): {put}\n")

    # --- Longstaff-Schwartz Monte Carlo ---
    iterations = 10000
    opt_type = "C"
    ls = LongstaffSchwartz(S0, K, T, t, r, r, sigma, iterations, 1000, opt_type, "Laguerre", 3)
    call = ls.option_price()
    opt_type = "P"
    ls = LongstaffSchwartz(S0, K, T, t, r, r, sigma, iterations, 1000, opt_type, "Laguerre", 3)
    put = ls.option_price()

    print(f"Call price (Longstaff-Schwartz): {call}")
    print(f"Put price (Longstaff-Schwartz): {put}\n")