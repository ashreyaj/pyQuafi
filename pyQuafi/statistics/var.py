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
    
class VaR_historical(VaR):
    def __init__(self, stocks, investment, confidence, nDays, start, end):
        super().__init__(stocks, investment, confidence, nDays, start, end)

    def var(self):
        data = self.returns()
        var_data = {}
        for s in self.stocks:
            mu = data[f'{s} Returns'].mean()
            sigma = data[f'{s} Returns'].std()
            var_data[s] = self.investment[s] * (mu * self.nDays - np.sqrt(self.nDays) * sigma * norm.ppf(1-self.confidence))
        return var_data
    
    def cvar(self):
        data = self.returns()
        var = self.var()
        cvar_data = {}
        for s in self.stocks:
            lt_var = data[f'{s} Returns'][data[f'{s} Returns'] < var[s]]
            cvar_data[s] = lt_var.mean()
        return cvar_data
    
class Var_MC(VaR):
    def __init__(self, stocks, investment, confidence, nDays, start, end, iterations):
        super().__init__(stocks, investment, confidence, nDays, start, end)
        self.iterations = iterations

    def var(self):
        data = self.returns()
        var_data = {}
        for s in self.stocks:
            mu = data[f'{s} Returns'].mean()
            sigma = data[f'{s} Returns'].std()
            rand = np.random.normal(0, 1, [1,self.iterations])
            S = investment[s] * np.exp(self.nDays * (mu - 0.5 * sigma**2) + sigma * np.sqrt(self.nDays) * rand)
            S = np.sort(S)
            percentile = np.percentile(S, (1 - self.confidence) * 100)
            var_data[s] = np.mean(S - percentile)
        return var_data
    
    def cvar(self):
        data = self.returns()
        var = self.var()
        cvar_data = {}
        for s in self.stocks:
            lt_var = data[f'{s} Returns'][data[f'{s} Returns'] < var[s]]
            cvar_data[s] = lt_var.mean()
        return cvar_data
    
if __name__ == '__main__':
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2024, 1, 1)
    stocks = ['AAPL','GE']
    investment = {'AAPL': 1e6, 'GE': 1e6}
    confidence = 0.95
    nDays = 1
    var_his = VaR_historical(stocks, investment, confidence, nDays, start, end)
    v = var_his.var()
    cv = var_his.cvar()
    print(f"Value at risk (Historical method): {v}")
    print(f"Conditional value at risk (Historical method): {cv}")

    iterations = 10000
    var_mc = Var_MC(stocks, investment, confidence, nDays, start, end, iterations)
    v = var_mc.var()
    cv = var_mc.cvar()
    print(f"Value at risk (Monte Carlo): {v}")
    print(f"Conditional value at risk (Monte Carlo): {cv}")