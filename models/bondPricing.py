import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bond:
    def __init__(self, r0, T, parValue, kappa=None, theta=None, sigma=None, dt=None, nRuns=None):
        self.r0 = r0
        self.T = T
        self.parValue = parValue
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.nRuns = nRuns

    # price of bond assuming constant interest rate and continuous compounding
    def constantIR(self):
        return self.parValue * np.exp(-self.r0 * self.T)
    
    # price of bond with interest rate modelled by the Vasicek model
    def vasicek(self):
        nSteps = int(self.T / self.dt)
        ratesAll = []

        for _ in range(self.nRuns):
            rates = np.zeros(nSteps)
            rates[0] = self.r0
            for i in range(1,nSteps):
                dr = - self.kappa * (rates[i-1] - self.theta) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
                rates[i] = rates[i-1] + dr
            ratesAll.append(rates)

        # integrate interest rates trajectory over time
        integralRates = np.sum(ratesAll,axis=1) * self.dt

        # calculate bond price
        discountFactor = np.exp(-integralRates)
        bondPrice = self.parValue * discountFactor

        return np.mean(bondPrice)
    
if __name__ == '__main__':
    r0 = 0.2
    T = 4
    parValue = 1000
    kappa = 1
    theta = 0.3
    sigma = 1e-2
    dt = 0.005
    nRuns = 100

    # bond price using Vasicek model
    bond = Bond(r0, T, parValue, kappa, theta, sigma, dt, nRuns)
    bondPrice = bond.vasicek()
    print(f"Bond price (Vasicek model): {bondPrice}")

    # bond price assuming constant interest rate
    bondPrice_constantIR = bond.constantIR()
    print(f"Bond price (constant interest rate): {bondPrice_constantIR}")