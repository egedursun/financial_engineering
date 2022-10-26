import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):
    data = {}
    ticker = yf.download(stock, start_date, end_date)
    data[stock] = ticker["Adj Close"]
    return pd.DataFrame(data)


# Calculating the value at risk for tomorrow
def calculate_var(position, c, mu, sigma):
    v = norm.ppf(1-c)
    var = position * (mu - sigma * v)
    return var


# Calculating the value at risk for tomorrow
def calculate_var_n(position, c, mu, sigma, n):
    v = norm.ppf(1-c)
    var = position * (mu * n - sigma * np.sqrt(n) * v)
    return var


if __name__ == "__main__":
    start = datetime.datetime(2014, 1, 1)
    end = datetime.datetime(2018, 1, 1)
    stock_data = download_data("C", start, end)

    stock_data["returns"] = np.log(stock_data["C"] / stock_data["C"].shift(1))
    stock_data = stock_data[1:]

    # Define the investment
    S = 1e6

    # confidence level
    c = 0.99

    # assuming the daily returns are normally distributed
    mu = np.mean(stock_data["returns"])
    sigma = np.std(stock_data["returns"])

    print("Value at risk (95 percent): $%0.2f" % calculate_var(S, 0.95, mu, sigma))
    print("Value at risk (95 percent): $%0.2f" % calculate_var_n(S, 0.95, mu, sigma, 252))
    print()
    print("Value at risk (99 percent): $%0.2f" % calculate_var(S, c, mu, sigma))
    print("Value at risk (99 percent): $%0.2f" % calculate_var_n(S, c, mu, sigma, 252))
