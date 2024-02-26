import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class CAPM:
    
    def __init__(self, stocks, start, end, riskFreeRate):
        self.data = None
        self.stocks = stocks
        self.start = start
        self.end = end
        self.rfr = riskFreeRate

    def get_data(self):
        data = {}
        for s in self.stocks:
            ticker = yf.download(s, start=self.start, end=self.end)
            data[s] = ticker['Adj Close'].resample('M').last() # monthly adjusted closing price
        return pd.DataFrame(data)
    
    def initialize(self):
        data = self.get_data()
        data['stock return'] = np.log(data[self.stocks[0]]/data[self.stocks[0]].shift(1))
        data['market return'] = np.log(data['^GSPC']/data['^GSPC'].shift(1))
        data.dropna(inplace=True)
        self.data = data

    def beta_formula(self):
        covariance = np.cov(self.data['stock return'], self.data['market return'])
        beta = covariance[1,0] / covariance[1,1]
        return beta

    def regression(self):
        beta, alpha = np.polyfit(self.data['market return'], self.data['stock return'], deg=1)
        expectedReturn = self.rfr + beta * (self.data['market return'] - self.rfr)
        return expectedReturn, beta
    
    def plot_regression(self):
        # raw data
        plt.scatter(self.data['market return'], self.data['stock return'], color='coral')
        # linear regression
        expectedReturn, beta = self.regression()
        plt.plot(self.data['market return'], expectedReturn, 'k', label=rf'LR with $\beta={beta.round(2)}$')
        plt.xlabel('Market return')
        plt.ylabel(self.stocks[0]+' return')
        plt.legend(frameon=0)
        plt.show()
    
if __name__ == '__main__':
    capm = CAPM(['AAPL', '^GSPC'], '2010-01-01', '2017-01-01', 0.05)
    capm.initialize()
    capm.plot_regression()