class ZeroCouponBond():

    def __init__(self, principal, maturity, interest_rate):
        # Principal Amount
        self.principal = principal

        # Time to Maturity
        self.maturity = maturity

        # Market Interest Rate
        self.interest_rate = interest_rate

    def calculate_present_value(self, x, n):
        return x / ((1 + self.interest_rate) ** n)

    def calculate_price(self):
        return self.calculate_present_value(self.principal, self.maturity)


if __name__ == "__main__":

    bond = ZeroCouponBond(principal=1000,
                          maturity=2,
                          interest_rate=0.04)
    print("Discounted (Present) Price of the Bond: $", bond.calculate_price())

