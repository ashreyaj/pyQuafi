import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize

class Markowitz:

    def __init__(self, stocks, start, end, tradingDays, numberPortfolios):
        self.stocks = stocks
        self.start = start
        self.end = end
        self.tradingDays = tradingDays
        self.numberPortfolios = numberPortfolios

    def get_data(self):
        data = {}
        for s in self.stocks:
            ticker = yf.Ticker(s)
            data[s] = ticker.history(start=start, end=end)['Close']
        return pd.DataFrame(data)

    def plot_price(self):
        self.get_data().plot()
        plt.show()

    def calc_logReturns(self):
        data = self.get_data()
        return np.log(data/data.shift(1))
    
    def riskReturn_portfolio(self, weights, returns):
        meanAnnualReturns = returns.mean() * self.tradingDays
        covAnnualReturns = returns.cov() * self.tradingDays

        expectedReturn = (meanAnnualReturns * weights).sum()
        expectedRisk = np.sqrt(np.dot(weights.T, np.dot(covAnnualReturns, weights)))
        return expectedReturn, expectedRisk, expectedReturn / expectedRisk

    def generate_portfolios(self):
        portfolioReturns = []
        portfolioRisks = []
        portfolioWeights = []

        returns = self.calc_logReturns()

        for _ in range(self.numberPortfolios):
            weights = np.random.rand(len(stocks))
            weights /= np.sum(weights)
            portfolioWeights.append(weights)
            pReturn, pRisk, _ = self.riskReturn_portfolio(weights, returns)
            portfolioReturns.append(pReturn)
            portfolioRisks.append(pRisk)

        portfolioReturns = np.array(portfolioReturns)
        portfolioRisks = np.array(portfolioRisks)
        sharpe = portfolioReturns / portfolioRisks

        return portfolioWeights, portfolioReturns, portfolioRisks, sharpe
    
    def optimize_sharpe(self, weights, returns):
        return -self.riskReturn_portfolio(weights, returns)[2]
    
    def optimal_portfolio(self, weights, returns):
        # flags for the optimizer
        constraint = {'type':'eq', 'fun': lambda x: x.sum() - 1}
        bounds = [(0,1) for _ in range(len(self.stocks))]
        opt = scipy.optimize.minimize(self.optimize_sharpe, weights[0], bounds=bounds, constraints=constraint, args=returns)
        bestWeights = opt['x'].round(3)
        bestReturn, bestRisk, bestSharpe = self.riskReturn_portfolio(bestWeights, returns)
        return bestReturn, bestRisk, bestSharpe

    def plot_portfolios(self, returns, risks, sharpe):
        plt.figure()
        plt.scatter(risks, returns, c=sharpe, marker='o', cmap='viridis')
        plt.xlabel('Portfolio volatility')
        plt.ylabel('Portfolio return')
        plt.colorbar(label='Sharpe ratio')

    def plot_best_portfolio(self, bestReturn, bestRisk):
        plt.plot(bestRisk, bestReturn, 'red', marker='*')
        plt.show()

if __name__ == '__main__':

    # --- Requisite data ---
    # stocks in the portfolio
    stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

    # history to consider in the model
    start = '2010-01-01'
    end = '2017-01-01'

    # average number of trading days in a year
    tradingDays = 252

    # number of sample portfolios to generate
    numberPortfolios = 10000

    # --- Markowitz model ---
    np.random.seed(1)
    M = Markowitz(stocks, start, end, tradingDays, numberPortfolios)

    # statistics related to each of the possible portfolios
    portfolioWeights, portfolioReturns, portfolioRisks, sharpe = M.generate_portfolios()
    bestReturn, bestRisk, bestSharpe = M.optimal_portfolio(portfolioWeights, M.calc_logReturns())
    fig = M.plot_portfolios(portfolioReturns, portfolioRisks, sharpe)
    M.plot_best_portfolio(bestReturn, bestRisk)