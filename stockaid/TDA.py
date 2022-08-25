
# Copyright 2022 Jesse Dutton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the implementation of API calls to TD Ameritrade for getting quotes,
   historical data, and option chains. The common use cases are included. The
   TDA API is capable of much that is not included here. Completeness is not a
   goal of this code. The functions themselves are plumbing, and will not be
   called by most users, so the implemented parts of the API are documented
   instead. These are usable by calling the cache's api() function.

   You will need to register (for free) at developer.tdameritrade.com in order
   to get an apikey, which you pass to the cache in your key_chain as the
   "TDA" key. When registering your app, set the Order Limit to 120. The
   callback URL is not used here, and you may pass https://127.0.0.1:8080 since
   it is a required field. The API calls implemented here are documented below
   with the fields that we expect. Up to date info on these APIs are available
   at developer.tdameritrade.com

       TDA quote     get a stock quote. Cached 60 seconds.
           symbol        is the ticker symbol to quote

       TDA history   get historical data for a stock. Cached 1 day. As an
                     example, to get twenty years of data with a value for
                     every day that the market was open, you would set
                     periodType to "year", period to 20, and frequencyType to
                     "daily". Beware, some of their historical data has gaps.
                     There are functions in stockaid.LSTM that can work around
                     any gaps in data.
           symbol        is the ticker symbol of the stock.
           periodType    is one of "day", "month", "year", or "ytd"
           period        is the number of periods to return
           frequencyType is the frequency of candles to return. Based on the
                         periodType, a different set of values are valid.
                             day:   minute
                             month: daily, weekly
                             year:  daily, weekly, monthly
                             ytd:   daily, weekly

       TDA chains    get quotes for option chains related to a stock. Cached 3
                     minutes
           symbol        is the ticker symbol of the underlying stock
           includeQuotes is a string that is either "TRUE" or "FALSE"
           range         is one of
                            "ITM" for in the money
                            "NTM" for near the money
                            "OTM" for out of the money
                            "SAK" for strikes above market
                            "SBK" for strikes below market
                            "SNK" for strikes near market
                            "ALL" for all
           fromDate      is a string date formatted using Date.isoformat()
           toDate        is a string date formatted using Date.isoformat()
           optionType    is one of
                             "S"   for standard
                             "NS"  for non-standard
                             "ALL" for all
"""

from .throttle import LazyTokenBucket
import json
import pandas as pd


# 52WkHigh 52WkLow askId askPrice askSize assetMainType assetSubType assetType
# bidId bidPrice bidSize bidTick closePrice cusip delayed description digits
# divAmount divDate divYield exchange exchangeName highPrice lastId lastPrice
# lastSize lowPrice marginable mark markChangeInDouble
# markPercentChangeInDouble nAV netChange netPercentChangeInDouble openPrice
# peRatio quoteTimeInLong realtimeEntitled regularMarketLastPrice
# regularMarketLastSize regularMarketNetChange
# regularMarketPercentChangeInDouble regularMarketTradeTimeInLong
# securityStatus shortable symbol totalVolume tradeTimeInLong volatility
def pandify_quote(resp):
    return pd.read_json(resp, orient='records', convert_dates=False)


# columns=['datetime','open','close','high','low','volume']
def pandify_history(resp):
    try:
        obj = json.loads(resp)
        if obj['empty']:
            return None
        candles = json.dumps(obj["candles"])
    except:
        return None
    df = pd.read_json(candles, orient='records', convert_dates=False)
    df = df.sort_values('datetime')
    return df


# internal function that flattens json
def _iter_options(df, ul, ul_last, options):
    for date in options:
        for strike in options[date]:
            for x in options[date][strike]:
                y = json.dumps({0:x})
                o = pd.read_json(y, orient='index')
                o['expDate'] = date
                o['strike'] = strike
                o['underlying'] = ul
                o['underlyingLast'] = ul_last
                if df is not None:
                    df = df.append(o)
                else:
                    df = o
    return df


# see docs for the option object in the response at:
# https://developer.tdameritrade.com/option-chains/apis/get/marketdata/chains
# to that object we add expDate, strike, underlying, and underlyingLast when
# we flatten. You should note that the documentation is wrong for the *Price
# fields, e.g. what the docs call 'askPrice' is actually just 'ask'.
def pandify_chains(resp):
    df = None
    try:
        obj = json.loads(resp)
        ul = obj['underlying']['symbol']
        ul_last = obj['underlying']['last']
        df = _iter_options(df, ul, ul_last, obj['callExpDateMap'])
        df = _iter_options(df, ul, ul_last, obj['putExpDateMap'])
    except:
        return None

    return df


def register_TDA(cache):
    throt = LazyTokenBucket(120)
    cache.register_provider('TDA',
                       'https://api.tdameritrade.com/v1/marketdata/', throt)
    cache.register_api('TDA', 'quote', '{symbol}/quotes', 'symbol',
                       pandify_quote, key_map={'apikey':'TDA'},
                       url_params=['symbol'], cache_secs=60)
    cache.register_api('TDA', 'history', '{symbol}/pricehistory', 'symbol',
                       pandify_history, key_map={'apikey':'TDA'},
                       url_params=['symbol'], cache_secs=86400,
                       data_params=['periodType','period','frequencyType'])
    cache.register_api('TDA', 'chains', 'chains', 'symbol', pandify_chains,
                       key_map={'apikey':'TDA'}, cache_secs=180,
                       data_params=['symbol','includeQuotes','range',
                                    'fromDate', 'toDate','optionType'])
