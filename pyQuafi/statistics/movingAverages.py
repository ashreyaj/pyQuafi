import pandas as pd
import yfinance as yf
import datetime as datetime
import matplotlib.pyplot as plt

class MovingAverage:
    def __init__(self, stock, start, end, period):
        self.stock = stock
        self.start = start
        self.end = end
        self.period = period

    def get_data(self):
        ticker = yf.download(self.stock, start=self.start, end=self.end)
        data = ticker['Adj Close']
        return data

    def sma(self):
        data = self.get_data()
        sma = data.rolling(window=self.period).mean()
        return sma
    
    def ema(self):
        data = self.get_data()
        ema = data.ewm(span=self.period, adjust=False).mean()
        return ema
    
if __name__=='__main__':
    ma = MovingAverage('AAPL', datetime.datetime(2017,1,1), datetime.datetime(2023,1,1), 200)
    stock_price = ma.get_data()
    stock_price.plot(label='Stock price')
    sma = ma.sma()
    sma.plot(label='SMA')
    ema = ma.ema()
    ema.plot(label='EMA')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    