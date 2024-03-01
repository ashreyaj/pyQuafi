import numpy as np

def simple(pv, r, t):
    """
    Args:
        pv (float): present value
        r (float): interest rate per time period
        t (int): number of time periods

    Returns: 
        future value
    """
    return pv * r * t

def compound(pv, r, t, m=1):
    """
    Args:
        pv (float): present value
        r (float): interest rate per time period
        t (int): number of time periods
        m (int): number of times interest applied per time period

    Returns: 
        future value
    """
    return pv * (1+(r/m))**(m*t)

def contcompound(pv, r, t):
    """
    Args: 
        pv (float): present value
        r (float): interest rate
        t (float): time

    Returns:
        future value
    """
    return pv * np.exp(r*t)