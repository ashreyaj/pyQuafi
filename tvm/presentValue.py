import numpy as np

def simple(fv, r, t):
    """
    Args:
        fv (float): future value
        r (float): interest rate per time period
        t (int): number of time periods

    Returns: 
        present value
    """
    return fv / (r*t)

def compound(fv, r, t, m=1):
    """
    Args:
        fv (float): future value
        r (float): interest rate
        t (int): number of time periods
        m (int): number of times interest applied per time period

    Returns: 
        present value
    """
    return fv / (1+(r/m))**(m*t)

def contcompound(fv, r, t):
    """
    Args: 
        fv (float): future value
        r (float): interest rate
        t (float): time

    Returns:
        present value
    """
    return fv * np.exp(-r*t)