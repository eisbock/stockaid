from enum import Enum
import numpy as np

class Future(Enum):
    FUTURE_DEFAULT = 0
    FUTURE_FIRST = 1
    FUTURE_MIN = 2
    FUTURE_MAX = 3
    FUTURE_LAST = 4

def guess_algo(val_col):
    """Attempt to find an appropriate algo based on column name"""
    if val_col == 'open':
        return Future.FUTURE_FIRST
    elif val_col == 'high':
        return Future.FUTURE_MAX
    elif val_col == 'low':
        return Future.FUTURE_MIN
    else:
        # treat unknown column names like 'close'
        return Future.FUTURE_LAST

def future_val(future, future_algo):
    """Extract a future value using a future_algo
       future must be a numpy array
    """
    if future_algo == Future.FUTURE_FIRST:
        return future[0]
    elif future_algo == Future.FUTURE_MAX:
        return np.amax(future)
    elif future_algo == Future.FUTURE_MIN:
        return np.amin(future)
    else:
        return future[-1]
