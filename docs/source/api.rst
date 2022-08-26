Stockaid
========

.. module:: stockaid
    :noindex:
    :synopsis: supported Stockaid APIs

This document describes the stockaid python module, including the thrid party
web APIs that it provides access to. Stockaid provides access to data related
to marketable securities in a way that is useful for use in machine learning.
It uses and depends on the :py:mod:`pandas` and :py:mod:`numpy` modules.

Key to this data access is the :class:`APICache`, which uses :py:mod:`requests`
to access registered web APIs, applies throttling to comply with the API's
policies, caches results locally, and returns the data in a
:py:class:`pandas.DataFrame`. Applications should use this class as a singleton
accessed through the :func:`get_cache` function.

.. autofunction:: stockaid.get_cache

APIs
----

Most commonly, you will use the :func:`APICache.api` function to remotely
access a third party API. A handful of these APIs are pre-registered with the
cache singleton, and those pre-registered APIs are documented here. It is also
possible to register new APIs using :class:`APICache`.

Each API is accessed using two strings, a provider and name. Currently, the
singleton has pre-registered providers called `index` and `TDA`. The project
roadmap calls for future support of an `EDGAR` provider, which will roll out
with our planned future support for sentiment analysis.

The `index` provider scrapes information from Wikipedia and caches the results
for one week. There are no parameters for these requests, so just the provider
and name arguments are required. The `index` provider supports these API names.

========== ========== ============================
Provider   Name       Description
========== ========== ============================
index      sp500      S&P 500 Index
index      OEX        S&P 100 Index
index      midcap     S&P MidCap 400 Index
index      smallcap   S&P SmallCap 600 Index
index      nasdaq100  Nasdaq 100
index      DJIA       Dow Jones Industrial Average
========== ========== ============================

TODO: TDA provider, automodule
