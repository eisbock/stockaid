"""Fetching and ML pre-processing of (free) stock data.
   Also included are utilities for interfacing with MiniZinc programs, and for
   using option Greeks to convert predictions about stock prices into
   predictions about option prices.
"""

__version__ = "0.1"

_stockaid_cache = None

from .cache import APICache
from .future_algo import Future
from .index import register_index
from .TDA import register_TDA
from .LSTM import LSTMHistory, register_LSTM
from . import greeks, MiniZinc, logging

def get_cache(cache_path=None, key_chain=None, mode=0o777):
    """Create the cache singleton, or return the existing one. The first time
       this is called, the arguments are used to create the cache. After the
       cache exists, these are not needed

           cache_path is the path to the cache directory to use
           key_chain  is a map from name to value of a list of api keys that
                      might be used in api calls.
           mode       is the mode to use when a file or dir needs creating.
    """

    global _stockaid_cache

    if not _stockaid_cache:
        _stockaid_cache = APICache(cache_path, key_chain, mode)
        register_index(_stockaid_cache)
        register_TDA(_stockaid_cache)
        register_LSTM(_stockaid_cache)

    return _stockaid_cache
