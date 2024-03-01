import numpy as np
import presentValue as pv

class ZeroCouponBond:
    """
    Args: 
    P (float): par value
    T (float): time to maturity
    r (float): risk-free interest rate
    """
    def __init__(self, P, T, r):
        self.P = P
        self.T = T
        self.r = r

    def fairBondPrice(self):
        return pv.compound(self.P, self.r, self.T)
    
class CouponBond:
    """
    Args: 
    P (float): par value
    T (float): time to maturity
    r (float): risk-free interest rate
    c (float): coupon rate
    m (int): number of coupon payments per annum
    """
    def __init__(self, P, T, r, c, m):
        self.P = P
        self.T = T
        self.r = r
        self.c = c
        self.m = m

    def couponValue(self):
        return self.c * self.P / self.m
    
    def effectiveIR(self):
        """
        Effective risk-free interest rate
        """
        return self.r / self.m
    
    def fairBondPrice(self):
        price = 0
        numberOfPayments = self.m * self.T
        # discounted coupons
        for i in range(1, numberOfPayments+1):
            price += pv.compound(self.couponValue(), self.effectiveIR(), i) 
        # discounted par value
        if self.couponValue()==0: # zero coupon bond
            price += pv.compound(self.P, self.r, self.T)
        else:
            price += pv.compound(self.P, self.effectiveIR(), numberOfPayments)

        return price
    
if __name__ == "__main__":
    P = 1000
    T = 7
    r = 0.1
    m = 5
    c = 0.0
    bond = ZeroCouponBond(P, T, r)
    cbond = CouponBond(P, T, r, c, m)
