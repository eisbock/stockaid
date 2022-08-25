"""Implementations of throttling functions."""

import time

class CrudeThrottler:
    """This throttler is crude. It makes calls_per_min calls then sleeps for
       a minute.
    """

    def __init__(self, calls_per_min):
        self.cpm = calls_per_min
        self.count = 0

    def throttle(self):
        self.count = self.count + 1
        if self.count > self.cpm:
            self.count = 0
            time.sleep(60)


class LazyTokenBucket:
    """This throttler is similar to the well known Token Bucket algorithm,
       except tokens are only added to the bucket when the throttle function
       is called, and all withdrawals are for one token. The last call time is
       stored to allow us to lazy-add the correct number of tokens. Sleeps in
       one second increments. Not a good fit for rates less than 60 calls per
       minute.
    """

    def __init__(self, calls_per_min):
        self.M = calls_per_min      # Max size of bucket
        self.b = self.M             # number in the bucket
        self.r = self.M / 60        # rate that tokens are added
        self.last = time.time()

    def throttle(self):
        # the lazy part -- catch the bucket up
        now = time.time()
        self.b = self.b + (self.r * (now - self.last))
        if self.b > self.M:
            self.b = self.M
        self.last = now

        # maybe sleep
        if self.b < 1:
            time.sleep(1)
            return self.throttle()

        # now deduct the token we are consuming
        self.b = self.b - 1

