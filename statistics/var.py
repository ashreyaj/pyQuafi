import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
import datetime

class VaR:
    def __init__(self, stocks, investment, confidence, nDays, start, end):
        self.stocks = stocks
        self.investment = investment
        self.confidence = confidence
        self.nDays = nDays
        self.start = start
        self.end = end
        self.data = None

    def get_data(self):
        data = {}
        for stock in self.stocks:
            ticker = yf.download(stock, self.start, self.end)
            data[stock] = ticker['Adj Close']
        return pd.DataFrame(data)
    
    def returns(self):
        data = self.get_data()
        for s in self.stocks:
            data[f'{s} Returns'] = np.log(data[s]/data[s].shift(1))
        data.dropna(inplace=True)
        return data
    
    def var_historical(self):
        data = self.returns()
        var_data = {}
        for s in self.stocks:
            mu = data[f'{s} Returns'].mean()
            sigma = data[f'{s} Returns'].std()
            var_data[s] = investment[s] * (mu * self.nDays - np.sqrt(self.nDays) * sigma * norm.ppf(1-self.confidence))
        return var_data
    
if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2024, 1, 1)
    stocks = ['AAPL','GE']
    investment = {'AAPL': 1e6, 'GE': 1e6}
    confidence = 0.95
    nDays = 1
    var = VaR(stocks, investment, confidence, nDays, start, end)
    v = var.var_historical()