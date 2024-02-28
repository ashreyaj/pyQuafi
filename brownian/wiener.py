import numpy as np
import matplotlib.pyplot as plt

class GeometricBrownianMotion:

    def __init__(self, dt, S0, drift, SDGaussian, n):
        """
        Args: 
        dt (float): time step
        S0 (float): initial asset price
        drift (float): drift in asset price
        SDGaussian (float): strength of price fluctuations
        n (int): number of integration steps
        """
        self.dt = dt
        self.S0 = S0
        self.drift = drift
        self.n = n
        self.SDGaussian = SDGaussian

    def wiener(self):
        W = np.zeros(self.n)
        W[0] = self.S0
        t = np.array(range(self.n)) * self.dt
        increments = np.random.normal(0, self.SDGaussian, self.n-1)
        W[1:self.n] = np.cumsum(increments)
        return t, W
    
    def gbm(self):
        _, W = self.wiener()
        t = np.array(range(self.n)) * self.dt
        X = (self.drift - 0.5 * self.SDGaussian**2) * t + self.SDGaussian * W
        S = self.S0 * np.exp(X)
        return t, S

if __name__ == '__main__':
    gbm = GeometricBrownianMotion(0.01, 1, 0.1, 0.1, 1000)
    time, returns = gbm.gbm()
    plt.plot(time,returns)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.show()