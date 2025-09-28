import numpy as np
import warnings

def check_naninf(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if np.isnan(result).any() or np.isinf(result).any():
            warnings.warn("Function returned NaN or Inf")
        return result
    return wrapper
