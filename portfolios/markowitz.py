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

    def statistics_portfolio(self):
        logReturns = self.calc_logReturns()

        # stock-wise historical mean and variance of annual returns
        meanAnnualReturns = logReturns.mean() * self.tradingDays
        covAnnualReturns = logReturns.cov() * self.tradingDays

        return meanAnnualReturns, covAnnualReturns

    def generate_portfolios(self):
        portfolioReturns = []
        portfolioRisks = []
        portfolioWeights = []

        meanAnnualReturns, covAnnualReturns = self.statistics_portfolio()

        for _ in range(self.numberPortfolios):
            weights = np.random.rand(len(stocks))
            weights /= np.sum(weights)
            portfolioWeights.append(weights)
            portfolioReturns.append((meanAnnualReturns * weights).sum())
            portfolioRisks.append(np.sqrt(np.dot(weights.T, np.dot(covAnnualReturns, weights))))

        portfolioReturns = np.array(portfolioReturns)
        portfolioRisks = np.array(portfolioRisks)
        sharpe = portfolioReturns / portfolioRisks

        return portfolioWeights, portfolioReturns, portfolioRisks, sharpe

    def plot_portfolios(self):
        _, returns, risks, sharpe = self.generate_portfolios()
        plt.figure()
        plt.scatter(risks, returns, c=sharpe, marker='o')
        plt.xlabel('Expected risk')
        plt.ylabel('Expected returns')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

if __name__ == '__main__':

    # stocks in the portfolio
    stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']

    # history to consider in the model
    start = '2010-01-01'
    end = '2017-01-01'

    # average number of trading days in a year
    tradingDays = 252

    # number of sample portfolios to generate
    numberPortfolios = 10000

    # run Markowitz
    np.random.seed(1)
    M = Markowitz(stocks, start, end, tradingDays, numberPortfolios)
    M.plot_portfolios()