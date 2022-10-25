import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

# Stocks we are going to handle
stocks = ["AAPL", "WMT", "TSLA","GE", "AMZN", "DB"]

# Historical data - define start and end dates
start_date = "2012-01-01"
end_date = "2017-01-01"


def download_data():
    # Name of the stock (key) and stock values
    stock_data = {}

    for stock in stocks:
        # closing prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date,
                                           end=end_date)["Close"]

    df = pd.DataFrame(stock_data)
    return df


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data):
    # The reason for log-return is applying "NORMALIZATION".
    log_return = np.log(data/data.shift(1))
    return log_return[1:]


def show_statistics(returns):
    # Instead of daily metrics, we want the yearly metrics

    # Mean of annual return
    print("Mean of Annual Returns: ", returns.mean() * NUM_TRADING_DAYS)

    # Covariance of daily returns
    print("Covariance of Annual Return: ", returns.cov() * NUM_TRADING_DAYS)


def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, (np.dot(returns.cov() * NUM_TRADING_DAYS, weights))))

    print("Expected Portfolio Return - Mean: ", portfolio_return)
    print("Expected Portfolio Volatility - STD: ", portfolio_volatility)


def generate_portfolio(returns):

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w * NUM_TRADING_DAYS))
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker="o")
    plt.grid()
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS,
                                                            weights)))

    return np.array([portfolio_return, portfolio_volatility,
                     portfolio_return / portfolio_volatility])


# Scipy optimize module can find the minimum of a given function
# The maximum of a function is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):
    constraints = {"type" : "eq",
                   "fun" : lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(stocks)))

    return optimization.minimize(min_function_sharpe, x0=weights[0], args=returns,
                          method="SLSQP", bounds=bounds, constraints=constraints)


def print_optimal_portfolio(optimum, returns):
    print("Optimal Portfolio: ", optimum["x"].round(3))
    print("Expected Return, Volatility, and Sharpe Ratio: ", statistics(optimum["x"].round(3),
                                                                        returns))


if __name__ == "__main__":
    dataset = download_data()
    show_data(dataset)
    log_daily_returns = calculate_return(dataset)
    # show_statistics(log_daily_returns)

    pweights, means, risks = generate_portfolio(log_daily_returns)
    show_portfolios(means, risks)

    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)

    show_optimal_portfolio(optimum, log_daily_returns,
                           means, risks)



