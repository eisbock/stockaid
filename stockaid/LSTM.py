
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

from datetime import datetime
import pandas as pd
import numpy as np
from .logging import log
from .future_algo import Future, guess_algo, future_val

_stockaid_cache = None


class LSTMHistory:
    """This class allows you to shape historic stock data for use in training
       an LSTM, which is a type of model used in machine learning. In an LSTM,
       the input is an array of data looking back over some number of days, and
       the output is a value for a single day. Each date's value is represented
       by a single float, so you specify which float column in a DataFrame is
       the value you will be training.

       The look_back period and look_ahead period are used to create each data
       point. The total set of data per stock might span a period much greater
       than the look_back. For example, if you want to train on the last 20
       years of daily stock data (in TDA terms, period=20, periodType='year',
       frequencyType='daily') it is not useful to stuff all of that data into
       a single data point. Instead, you might wish to define a look_back that
       spans the prior quarter (or look_back=65 days). Each starting date in
       the historical data will be transformed into a data point where an
       array of the previous look_back dates forms the X and the forward
       look_ahead dates (numbered from 1) are used to form the Y used to train
       the ML. If look_ahead == 1, it is the day following that last day in
       the look_back. You would use look_ahead == 1 to predict tomorrow's
       value.

       As with all ML, the quality of the data you put in will affect the
       utility of the model. Data must be scaled to allow the underlying math
       to converge. In trials, training on a single stock was not yielding
       specific enough predictions for my liking, so this class was designed
       for data that spans different stocks, although you might also use it to
       ingest only one stock. In addition, some stocks have a wide range of
       values across their history, so scaling their entire history was also
       problematic. As a result, this class will scale each data point
       (historical look_back with a future value) individually by dividing out
       the mean from the floats in that data point. Since this is a linear
       modification, the percentages used to evaluate the test data are not
       altered, so the scaling factors for the data points used to train do not
       need to be stored. By default, data more than 5.0 times the mean will
       not be included in the fit or test sets. You can adjust this parameter
       (called clip) as you see fit.

       Some of the data in the set will not have a future value to train on.
       These are the data points where the look_ahead range is partially or
       entirely in the future. This class calls those data points future data.
       These future data are tagged (with a string, perhaps the symbol name)
       when ingested so that you can predict one stock at a time. The scaling
       factor used to scale the future data points is stored using the same
       tag, and predictions can be unscaled using the the unscale_future
       function. Depending on the column you choose to train on, selecting a
       value from the look_ahead will use different methods. For example, when
       training on close, the last close value in the look_ahead is needed, but
       when training on high, the max(high) value over the look_ahead is what
       you want. For data from TDA history, reasonable choices are made based
       on the column name, but you can override this algorithm in the ingest
       function, if you are sourcing your data elsewhere.

       Conveinence functions are provided to ingest data based on symbol or
       index name. These functions use the TDA history api, so you will need
       to have supplied an api key for TDA when you created the cache, if you
       use those functions.
    """

    def __init__(self, look_back, look_ahead, test_ratio=0.2, will_train=True,
                 will_predict=True, future_history=0):
        """If you are training an LSTM to predict stock prices based on
           historic values, use this class to wrangle data. This works well
           with the TDA historic api registered with the cache.

           :param look_back: the number of dates in the past to train on.
           :param look_ahead: the offset to the end date for the set of future
               data being predicted on.
           :param test_ratio: the (decimal) percent of data to use for testing.
           :param will_train: can be set to False to reduce memory usage if you
               will not be training on this data.
           :param will_predict: can be set to False to reduce memory usage if
               you will not be predicting on the future values.
           :param future_history: the number of look_back sets to include that
               we know the future for already. Defaults to 0. Use 1 if you wish
               to calibrate or filter future predictions based on how far off
               the prediction is from the last value. See last_val().
           :raises ValueError: if look_back, look_ahead, or test_ratio are out
               of range.
        """
        if look_back <= 0:
            raise ValueError("look_back must be > 0")
        if look_ahead <= 0:
            raise ValueError("look_ahead must be > 0")
        if test_ratio < 0 or test_ratio >= 1:
            raise ValueError("test_ratio must be >= 0")

        self.look_back = look_back
        self.look_ahead = look_ahead
        self.test_ratio = test_ratio
        self.keep_train = will_train
        self.keep_future = will_predict
        self.future_history = future_history
        self.count_fit = 0
        self.count_test = 0
        self.fit_x = np.array([])
        self.fit_y = np.array([])
        self.test_x = np.array([])
        self.test_y = np.array([])
        self.future = {}
        self.scale = {}
        self.lval = {}


    def ingest(self, df, tag, date_col='datetime', val_col='close',
               future_algo=Future.FUTURE_DEFAULT, clip=5.0):
        """
           Adds a pandas DataFrame to the data set, along with a tag. If this
           function is called multiple times, the multiple DataFrames are added
           to the fit, test, and future sets, but the future data is tagged so
           that prediction can happen on data from a single DataFrame, even if
           training spans multiple tagged sets.

           :param df: the pandas DataFrame that holds the data. The data is
               copied, so you may reuse your DataFrame after this call returns.
           :param tag: a string used to identify the data (e.g. the symbol).
           :param date_col: the name of the df column that has the date. This
               column is used to sort the data.
           :param val_col: the name of the df column that has the float value
               to train the model to predict.
           :param future_algo: describes how a future value should be chosen.
               For use with TDA history, the val_col will map to an appropriate
               future_algo automatically. To make a manual choice, choose from
               the Future enum values.
           :param clip: the limit for valid values in training data. After
               scaling, sets that contain a value above this are omitted. Since
               scaling divides by the mean, the default means that the lookback
               period contains a value that is 5x the mean.
           :raises ValueError: if tag is missing, if either date_col or val_col
               are not valid, or if the data set has too few rows to
               accommodate the look_back and look_ahead.
        """
        if not tag:
            raise ValueError("a tag is required")
        if df is None or not set([date_col, val_col]).issubset(df.columns):
            raise ValueError("column names not in DataFrame {}[{},{}]".format(
                             tag, date_col, val_col))
        lb = self.look_back
        la = self.look_ahead
        if len(df) < lb + la:
            raise ValueError("DataFrame {} has too little data".format(tag))

        # reasonable defaults
        if future_algo == Future.FUTURE_DEFAULT:
            future_algo = guess_algo(val_col)

        # copy, sort (oldest first), and index
        df = df.copy()
        df = df.sort_values(date_col)

        # decision variable for which set the row is added to
        split = np.random.random(len(df))

        # data as a one dimensional ndarray of floats from val_col
        s = np.array(df[val_col], dtype=float).flatten()
        self.lval[tag] = s[-1]

        # training data
        if self.keep_train:
            # numpy append does not scale well, so build small arrays and
            # append just once instead of in a loop
            tx = []
            ty = []
            fx = []
            fy = []
            for i in range(lb, len(s) - la):
                # scale each history / future independent of the others
                x = s[i-lb:i+la]    # mixed data, history and future
                m = np.mean(x)
                x = x / m
                if np.amax(x) > clip:
                    continue
                y = future_val(x[lb:], future_algo)
                x = x[:lb]
                if split[i] < self.test_ratio:
                    tx.append(x)
                    ty.append(y)
                    self.count_test = self.count_test + 1
                else:
                    fx.append(x)
                    fy.append(y)
                    self.count_fit = self.count_fit + 1
            self.test_x = np.append(self.test_x,tx).reshape(self.count_test,lb)
            self.test_y = np.append(self.test_y,ty)
            self.fit_x = np.append(self.fit_x,fx).reshape(self.count_fit,lb)
            self.fit_y = np.append(self.fit_y,fy)

        # future data
        if self.keep_future:
            # scale the data
            start = len(s) - la -lb - self.future_history
            if start < 0:
                start = 0
            x = s[start:]
            self.scale[tag] = np.mean(x)
            x = x / self.scale[tag]

            # now reshape the data as a series of lb-sized histories
            f = np.array([])
            for i in range(lb, len(x)):
                f = np.append(f, x[i-lb:i])
            self.future[tag] = f.reshape(len(x)-lb, lb)


    def ingest_symbol(self, symbol, period, ptype, freq_type, val_col='close',
                      clip=5.0):
        """Fetch a stock symbol using TDA history, then ingest. If the cache
           needs to refresh this historical data, it will require the TDA api
           key. See the get_cache() function for more details.

           :param symbol: the stock symbol, becomes the tag during ingest().
           :param period: the period argument to TDA history.
           :param ptype: the periodType argument to TDA history.
           :param freq_type: the frequencyType argument to TDA history.
           :param val_col: passed through to ingest().
           :param clip: passed through to ingest().
           :raises ValueError: if the val_col is not valid or the data set has
               too few rows to accommodate the look_back and look_ahead.
        """
        global _stockaid_cache
        hist = _stockaid_cache.api('TDA', 'history', symbol=symbol,
                                   periodType=ptype, period=period,
                                   frequencyType=freq_type)
        self.ingest(hist, symbol, val_col=val_col, clip=clip)


    def ingest_index(self, index, period, ptype, freq_type, val_col='close',
                     clip=5.0, omit=[], quiet=False):
        """Fetch each stock symbol in a given index registered with the cache.
           The stock histories are fetched using the TDA history api. During
           ingest(), the symbol name is used as the tag. If the cache needs to
           refresh any historical data, it will require the TDA api key. See
           the get_cache() function for more details.

           :param index: an index name registered by the index api provider.
           :param period: the period argument to TDA history.
           :param ptype: the periodType argument to TDA history.
           :param freq_type: the frequencyType argument to TDA history.
           :param val_col: passed through to ingest().
           :param clip: passed through to ingest().
           :param omit: if provided, is a list of symbols in the index to omit
           :param quiet: if True, will suppress log messages related to each
               stock that is ingested. You can also set the log level less than
               2 (info) to suppress these messages.
           :raises ValueError: if the val_col is not valid or the data set has
               too few rows to accommodate the look_back and look_ahead.
        """
        global _stockaid_cache
        ts = datetime.now().timestamp()
        idx_list = _stockaid_cache.api('index',index)['Symbol']
        for symbol in idx_list:
            if symbol in omit:
                continue
            self.ingest_symbol(symbol, period, ptype, freq_type, val_col, clip)
            if not quiet:
                end = datetime.now().timestamp()
                log(2, "Ingest {} in {:.3f} seconds".format(symbol,end-ts))
                ts=end


    def fit_data(self):
        """When training, the fitness function will need X, Y data that
           contains the look_back and look_ahead values, respectively.

           :returns: two numpy ndarrays representing the X and Y data for your
               model's fit function.
        """
        return self.fit_x, self.fit_y

    def test_data(self):
        """When evaluating your model, the reserved test data is needed. After
           running predictions on this data, you will need to run the
           score_test function to evaluate the performace of the model.

           :returns: a numpy ndarray containing the X data for your model's
               predict function.
        """
        return self.test_x

    def score_test(self, predictions):
        """After fitting your model, as a test, you should predict on the
           test_data(). This function compares that predicted data to the true
           values.

           :param predictions: an array-like (i.e. castable to an ndarray) set
               of floats that represent the output of your model.
           :returns: two metrics, MAPE and MPE. The Mean Absolute Percentage
               Error is a positive percent (float) that evaluates performance,
               with a smaller number being better. MPE is the non-absolute
               version, which detects systematic errors in the model. For
               instance, if your MAPE and MPE are the same size, it indicates
               that your model always misses in the same direction (negative,
               if MPE is negative).
        """
        t = self.test_y
        p = np.array(predictions).reshape(t.shape)
        mape = np.mean(np.abs((p-t)/t))
        mpe = np.mean((p-t)/t)
        log(2, "ML test score: MAPE={:.4f}, MPE={:.4f}".format(mape,mpe))

        return mape, mpe

    def future_data(self, tag):
        """Since the model needs to look_ahead a certain number of dates, the
           newest sets will not have these look_ahead values. These new values
           are future data that you can access here, sorted by date, so the
           newest value is index -1 of the ndarray that is returned. After
           running your predict function on this data, you will need to run
           the unscale_future() function.

           :param tag: the tag provided when the data was ingested, or the
               symbol name if ingest_symbol or ingest_index was used.
           :returns: an ndarray with shape=(rows,look_back) of floats that
               contains the X parameter for your model's predict function, or
               None if tag is not valid.
        """
        return self.future.get(tag)

    def unscale_future(self, tag, predictions):
        """This function will unscale a set of predictions. You should call
           it after model prediction on the future_data() data.

           :param tag: the tag provided when the data was ingested, or the
               symbol name if ingest_symbol or ingest_index was used.
           :param predictions: an array-like (i.e. castable to an ndarray) set
               of floats that represent the output of your model.
           :returns: an ndarray of floats that contains the unscaled predicted
               future values, or None if tag is not valid.
        """
        if self.scale.get(tag) is None:
            return None
        return np.array(predictions * self.scale[tag]).flatten()


    def last_val(self, tag):
        """The last value in the ingested data is stored as a conveinence. This
           function will return that float value. If you use future_history=1
           when creating this class, this gives you something to compare the
           predictive abilities of your model to on a stock by stock basis.

           :param tag: the tag provided when the data was ingested, or the
               symbol name if ingest_symbol or ingest_index was used.
           :returns: a float, the true last value in val_col, or None if tag
               is not valid.
        """
        return self.lval.get(tag)


    def iter_tags(self):
        """This function generates the set of tags ingested"""
        for t in self.future.keys():
            yield t

def register_LSTM(cache):
    global _stockaid_cache
    _stockaid_cache = cache
