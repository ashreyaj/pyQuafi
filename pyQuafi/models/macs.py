import datetime as datetime
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class MovingAverageCrossoverStrategy:
    def __init__(self, stock, start, end, investment, fast_period, slow_period):
        self.stock = stock
        self.start = start
        self.end = end
        self.investment = investment
        self.equity = [investment]
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.is_long = False
        self.data = None

    def get_data(self):
        data = {}
        ticker = yf.download(self.stock, start=self.start, end=self.end)
        data['Price'] = ticker['Adj Close']
        self.data = pd.DataFrame(data)

    def ma(self):
        self.get_data()
        self.data['MA_slow'] = self.data['Price'].ewm(span=self.slow_period).mean()
        self.data['MA_fast'] = self.data['Price'].ewm(span=self.fast_period).mean()
        return self.data
    
    def strategy(self):
        buy_price = 0
        for i,r in self.data.iterrows():
            # open long position when slow MA is below the fast MA
            if r['MA_slow'] < r['MA_fast'] and not self.is_long:
                buy_price = r['Price']
                self.is_long = True
            # close long position when slow MA is above fast MA
            elif r['MA_slow'] > r['MA_fast'] and self.is_long: 
                self.equity.append(self.investment * r['Price'] / buy_price)
                self.investment = self.investment * r['Price'] / buy_price
                self.is_long = False
        return np.array(self.equity)

if __name__ == '__main__':
    start = datetime.datetime(2010,1,1)
    end = datetime.datetime(2020,1,1)
    investment = 100
    fast_period = 30
    slow_period = 100
    macs = MovingAverageCrossoverStrategy('IBM', start, end, investment, fast_period, slow_period)
    d=macs.ma()
    strat = macs.strategy()
    plt.plot(strat/investment-1,'.-')
    plt.axhline(y=0,color='grey',ls='--')
    plt.xlabel('Time')
    plt.ylabel('Profit / Investment')
    plt.show()