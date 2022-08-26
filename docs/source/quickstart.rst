Quickstart
==========

This is the quick start guide. You are here because you need to guess at stock
prices based on historical data. First, we will get you access to raw data.
Next, we will build a toy ML model and feed it data using stockaid.

Install Stockaid
----------------

Use pip.

.. code-block:: text
    $ python3 -m pip install stockaid

Getting an API Key
------------------
Stockaid can access historic data from TDA Ameritrade. They provide data for
free, but you are required to register your app, which you can do
`here <https://developer.tdameritrade.com/>`_. During registration, they will
ask for a rate limit (120) and a callback URL. The callback URL is required by
the parts of the API that allow you to access account information on behalf of
a user. Stockaid does not use these parts of the API, and it is possible to
use an https link to 127.0.0.1 in this field. After your app is registered,
they will give you an API key. Save that somewhere, and add it to your shell.

.. code-block:: text
    $ export TDA_APIKEY='whatever TDA gave you'

The Smallest Amount of Code Possible
------------------------------------
Stockaid is for wrangling stock data, and really doesn't care how you use that
data in your ML. The example below uses Keras, because it is easy, but you are
free to make a different choice and still use stockaid. This toy model uses a
look back period of 20 days and is designed to guess at tomorrow's stock
prices.

.. code-block:: python
    import os
    import stockaid
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow is noisy by default
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense

    # get 20 years of data for the S&P 500 from TDA using stockaid
    lb=20
    cache = stockaid.get_cache(
                cache_path=os.path.expanduser('~/.stockaid'),
                key_chain={'TDA': os.environ.get('TDA_APIKEY')})
    data = stockaid.LSTMHistory(look_back=lb, look_ahead=1)
    data.ingest_index('sp500', 20, 'year', 'daily') # 20 years of daily data

    # toy model
    inp = Input(shape=(lb, 1))
    out = LSTM(units=lb, return_sequences=True)(inp)
    out = LSTM(units=lb)(out)
    out = Dense(1, activation='linear')(out)
    model = Model(inp,out)
    model.compile(loss = 'mean_squared_error', optimizer='Nadam')

    # train the model
    x,y = data.fit_data()
    model.fit(x,y, epochs=10, batch_size=64, validation_split=0.1,
              shuffle=True)
    data.score_test(model.predict(data.test_data()))  # logs results

    # now guess at tomorrow's closing price for GOOG
    pred = model.predict(data.future_data('GOOG'))
    pred = data.unscale_future('GOOG', pred)
    tomorrow = pred[-1]  # the last date's value is at the end of the array
    print('Tomorrow, GOOG should close around {:.2f}'.format(tomorrow))

???, Profit
-----------
Notice that this model has been described as a toy and its predictions have
been described as guesses. The step that this library does for you is the data
wrangling. If you need a reliable ML model, you need to make a reliable ML
model. That being said, even with an amazing model, predicting the future is
likely to have unpredicatble outcomes. You might lose your shirt in the market.
Profits not guaranteed.

