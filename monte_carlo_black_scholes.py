import numpy as np

class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        # 2 columns. first 0s, second payoff
        # pay-off is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dim array with as many items
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.T) * rand
        )

        # calculate the max(S-E, 0)
        option_data[:, 1] = stock_price - self.E

        # average for the monte-carlo simulation
        # max() returns the max(0, S-E)
        average = np.sum(np.amax(
            option_data, axis=1
        )) / float(self.iterations)

        # have to use discount factor  exp(-rT)
        return np.exp(-1.0 * self.rf * self.T) * average


    def put_option_simulation(self):
        # 2 columns. first 0s, second payoff
        # pay-off is max(0, S-E) for call option
        option_data = np.zeros([self.iterations, 2])

        # dimensions: 1 dim array with as many items
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.T) * rand
        )

        # calculate the max(E-S, 0)
        option_data[:, 1] = self.E - stock_price

        # average for the monte-carlo simulation
        # max() returns the max(0, S-E)
        average = np.sum(np.amax(
            option_data, axis=1
        )) / float(self.iterations)

        # have to use discount factor  exp(-rT)
        return np.exp(-1.0 * self.rf * self.T) * average


if __name__ == "__main__":
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 1000)
    print("Value of the call option : $", round(model.call_option_simulation(), 2))
    print("Value of the put option: $", round(model.put_option_simulation(), 2))
