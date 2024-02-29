import numpy as np
import matplotlib.pyplot as plt

class Vasicek:

    def __init__(self, r0, kappa, theta, sigma, iterations, dt):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.iterations = iterations
        self.dt = dt

    def vasicek(self):
        rates = np.zeros(self.iterations)
        rates[0] = self.r0

        for i in range(1,self.iterations):
            dr = - self.kappa * (rates[i-1] - self.theta) + self.sigma * np.sqrt(self.dt) * np.random.normal()
            rates[i] = rates[i-1] + dr

        t = np.array(range(self.iterations)) * self.dt
        return t, rates
    
if __name__ == '__main__':
    r0 = 1.3
    kappa = 0.9
    theta = 1.5
    sigma = 0.1
    iterations = 1000
    dt = 0.01
    vasicek = Vasicek(r0, kappa, theta, sigma, iterations, dt)
    times, rates = vasicek.vasicek()
    plt.plot(times, rates)
    plt.axhline(y=theta, color='k', ls='--', label=r'$\Theta$')
    plt.xlabel('Time')
    plt.ylabel('Interest rate')
    plt.legend(frameon=0)
    plt.show()