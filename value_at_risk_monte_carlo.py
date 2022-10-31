import numpy as np
import yfinance as yf
import datetime
import pandas as pd


def download_data(stock, start, end):
    data = {}
    ticker = yf.download(stock, start, end)
    data["Adj Close"] = ticker["Adj Close"]
    return pd.DataFrame(data)


class ValueAtRiskMonteCarlo:

    n = 1

    def __init__(self, S, mu, sigma, c, n, iterations):
        # this is the value of our investment at t=0 (1000 usd)
        self.S = S
        # mean
        self.mu = mu
        # standard deviation
        self.sigma = sigma
        # confidence level
        self.c = c
        # time
        self.n = n
        # iterations
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price
        # the random walk of our initial investment
        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * self.sigma ** 2) +
            self.sigma * np.sqrt(self.n) * rand
        )

        # sort the prices to determine the percentile
        stock_price = np.sort(stock_price)

        # calculate percentile 95% percentile -> 5 , 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


if __name__ == "__main__":
    S = 1e6  # the initial investment
    c = 0.95  # confidence level
    n = 1  # 1 day
    iterations = 100000  # number of paths in the monte-carlo sim.

    # historical data to approximate mean and std
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2017, 10, 15)

    citi = download_data("C", start, end)

    # calculate daily returns
    citi["returns"] = citi["Adj Close"].pct_change()

    # assume returns are normally distributed:
    # mean and variance can describe the process
    mu = np.mean(citi["returns"])
    sigma = np.std(citi["returns"])

    model = ValueAtRiskMonteCarlo(S=S, mu=mu,
                                  sigma=sigma, 
                                  c=c, n=n,
                                  iterations=iterations)
    print("Value at Risk with Monte-Carlo: $", model.simulation())