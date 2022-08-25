
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

"""A collection of utils for applying the option Greeks to our predictions
   about the future. Specifically, we can look at how the price of an option
   is expected to change based on a predicted change in the price of the
   underlying security over some time period, a predicted change in volatility,
   or a predicted change in interest rates.
"""

def option_change(underlying_change, days_in_future, delta, gamma, theta):
    """It is common to close an option position without exercising. In those
       cases, the change in the underlying security does not equal the change
       in the price of the option. The Greeks are an attempt to predict these
       changes, and are included with an options quote, typically. However,
       the future prices predicted by the Greeks may not reflect reality
       perfectly, and are more likely to have error as days_in_future or
       underlying_change becomes larger.

           underlying_change is the difference between the predicted price
                             of the underlying security at days_in_future
                             and the current price.
           days_in_future    is the number of days forward being predicted
           delta             is the price change per $1 of underlying_change
           gamma             is how much delta changes per $1 of underlying
           theta             is the decay in option value per day

    The other common Greeks are rho and vega, and are not included in this
    function, since using them predictively would also require predictions
    about changes to interest rates and volatility, respectively.
    """

    underlying_change=float(underlying_change)
    days_in_future=int(days_in_future)
    delta=float(delta)
    gamma=float(gamma)
    theta=float(theta)

    # sanity check the greeks
    if delta < -1 or delta > 1 or gamma < 0 or gamma > 1:
        raise ValueError("Greeks out of bounds!")

    change = 0.0
    sign = 1
    if underlying_change < 0:
        sign = -1
    # whole dollars
    while abs(underlying_change) > 1:
        change = change + (sign * delta)
        underlying_change = underlying_change - sign
        delta = delta + (sign * gamma) # gamma is always positive
        # abs(delta) can never be > 1. This limit seeks to hedge against errors
        # introduced over a large number of days. In reality, gamma will also
        # adjust over time so that this limit is not actually needed when using
        # current values -- we are predicting the future. The rate at which
        # gamma changes over time (i.e. the third derivative of the price of
        # the underlying) is not a published Greek. That change is usually
        # insignificant over small changes to the price of the underlying.
        delta = min(delta, 1)
        delta = max(delta, -1)
    # leftover fraction
    change = change + (delta * underlying_change)

    # time decay: this is usually accurate for a small number of days. The
    # Greek theta also changes over time in a way that is not captured here.
    change = change + (days_in_future * theta)

    return change


def adjust_for_vega(predicted_volatility_change, vega):
    """Option prices can change in response to a change in volatility over some
       time period. The time period does not matter, as long as your prediction
       expresses the change in volatility over that period.

           predicted_volatility_change is the percentage change in volatility,
                                       expressed as a decimal
           vega                        is the option's value of this Greek
    """

    return predicted_volatility_change * 100 * vega


def adjust_for_rho(predicted_interest_rate_change, rho):
    """Option prices can change in response to a change in interest rates.
       This is often a prediction about how the Federal Reserve will adjust
       interest rates.

           predicted_interest_rate_change is the change in interest rates,
                                          expressed as a decimal
           rho                            is the option's value of this Greek
    """

    return predicted_interest_rate_change * 100 * rho


def option_lambda(underlying_price, option_price, delta):
    """Lambda is a measure of the leverage provided by an option. It is the
       ratio of the change in option price to the change in stock price. If you
       calculate a lambda of 50, then you would expect a 50% increase in option
       price for each 1% increase in stock price. It is not always present in
       an option quote, so this function is provided to calculate it.

       Beware that the value of delta changes in response to movement in the
       underlying, and delta is used in this calculation, so this is a
       reasonable estimate for small moves, but is error-prone if the stock
       price moves quite dramatically. Of course, a dramatic move is either
       quite profitable of bankruptcy-inducing, depending on which side you are
       on, so maybe leverage matters less in those cases.
    """

    return delta * underlying_price / option_price

