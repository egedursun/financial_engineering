from math import exp

class CouponBond:

    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        self.rate = rate
        self.maturity = maturity
        self.interest_rate = interest_rate

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n

    def cont_present_value(self, x, n):
        return x * exp(-self.interest_rate * n)

    def calculate_price(self, continuous=False):
        price = 0

        if continuous:

            # Discount the coupon payments
            for t in range(1, self.maturity + 1):
                res = self.principal * self.rate
                price += self.cont_present_value(res, t)

            # Discount the present amount
            price += self.cont_present_value(self.principal,
                                        self.maturity)

        else:

            # Discount the coupon payments
            for t in range(1, self.maturity + 1):
                res = self.principal * self.rate
                price += self.present_value(res, t)

            # Discount the present amount
            price += self.present_value(self.principal,
                                        self.maturity)

        return price


if __name__ == "__main__":

    bond = CouponBond(principal=1000, rate=0.1,
                      maturity=3, interest_rate=0.04)

    print("Present (Discounted) Value of the Bond: $", bond.calculate_price())
    print("Present Continuous Value of the Bond: $", bond.calculate_price(continuous=True))
