
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

"""This is the implementation of the caching, throttled, api client."""

import os
import time
from datetime import datetime
import requests
import pandas as pd

# returns False if the dir exists and is not writable, or if the dir cannot
# be created with a writable mode.
def _check_cache_dir(path,  mode):
    if os.path.exists(path):
        if os.path.isdir(path) and os.access(path, os.W_OK):
            return True
        return False
    os.mkdir(path, mode)
    return os.access(path, os.W_OK)


class APICache:
    """This class implements the caching, throttled, api cache. Normally, a
       singleton is used, which can be accessed with the function
       stockaid.get_cache()
    """

    def __init__(self, cache_path=None, key_chain=None, mode=0o777):
        self.cache_path = cache_path
        self.key_chain = key_chain
        self.mode = mode
        self.providers = {}
        if not _check_cache_dir(cache_path, self.mode):
            self.cache_path = None

    def can_cache(self):
        """returns a boolean that is True if the cache on disk is writable"""

        if self.cache_path:
            return True
        return False

    class _Provider:
        def __init__(self, base_url, cache_dir, throttler):
            self.base_url = base_url
            self.throttler = throttler
            self.cache_dir = cache_dir
            self.api_list = {}

        def register_api(self, name, t):
            self.api_list[name] = t

        def get_api(self, name):
            if name not in self.api_list:
                raise ValueError("API '{}' not registered.".format(name))
            return self.api_list[name]

        def throttle(self):
            if self.throttler:
                self.throttler.throttle()


    def register_provider(self, name, base_url, throttler=None):
        """Register a provider of an API. Calls to this provider respect the
           policy described by throttler.

               name      is a unique name for this provider
               base_url  is the common part of the url for all api calls
               throttler is an instance of a class from stockaid.throttle
        """

        cache_dir=None
        if self.cache_path:
            cache_dir = os.path.join(self.cache_path, name)
            if not _check_cache_dir(cache_dir, self.mode):
                cache_dir = None
        self.providers[name] = self._Provider(base_url, cache_dir, throttler)


    def register_api(self, provider, name, url, cache_field, pandify_fn,
            method='GET', url_params=[], data = None, data_params=[],
            key_map={}, cache_secs=0):
        """Register an API call of a provider. Arguments describe how to call
           the API, and provide a pandify_fn that converts the returned json
           into pandas DataFrame.

               provider    is the name of the provider previously registered
               name        is a unique name for this api call
               url         is the part of the url after the provider's base_url
               cache_field is the field to use to identify this request in the
                           cache.
               pandify_fn  is the function that converts the reponse text into
                           a pandas DataFrame. Return None on error.
               method      is passed to requests.request. Usually GET or POST.
               url_params  is a list of fields that should be used to format
                           the url.
               data        is the format string to use if the body of the post
                           must be json. In that case, data_params will be
                           substituted into this string using format.
               data_params is a list of fields that should either be passed as
                           parameters or substituted into the json if provided
                           by the data argument.
               key_map     is a map from field name to key name. These will be
                           merged with the params sent to the server, but these
                           fields are not passed as arguments. Instead, they
                           were passed in when the APICache was created. The
                           intended use is for api keys that should not be
                           stored in source control. This allows you to wrangle
                           these at the start of your program and mostly ignore
                           them after.
               cache_secs  is the default number of seconds that calls to this
                           api should be cached. 0 means no caching.
        """

        if provider not in self.providers:
            raise ValueError("Provider '{}' not registered.".format(provider))

        cache_dir = None
        if self.providers[provider].cache_dir:
            cache_dir = os.path.join(self.providers[provider].cache_dir, name)
            if not _check_cache_dir(cache_dir, self.mode):
                cache_dir = None

        if self.providers[provider].base_url:
            api_url = '{}/{}'.format(self.providers[provider].base_url, url)
        else:
            api_url = url

        t = {'url':api_url, 'method':method, 'cache_field':cache_field,
                'pandify_fn':pandify_fn, 'url_params':url_params, 'data':data,
                'data_params':data_params, 'key_map':key_map,
                'cache_secs':cache_secs, 'cache_dir':cache_dir}
        self.providers[provider].register_api(name, t)


    def api(self, provider, name, refresh=False, **kwargs):
        """Call a registered API. Returns a pandas DataFrame, or None on error.

               provider is the registered name of the provider
               name     is the registered name of the api
               refresh  is a boolean. True will ignore cached results.
               **kwargs are the set of fields that are specific to this api
        """
        if provider not in self.providers:
            raise ValueError("Provider '{}' not registered.".format(provider))
        t = self.providers[provider].get_api(name)

        # see if the cached copy is still valid
        cache_file = None
        if t['cache_dir']:
            if t['cache_field']:
                cache_file = os.path.join(t['cache_dir'],
                        '{}.csv'.format(kwargs[t['cache_field']]))
            else:
                cache_file = os.path.join(t['cache_dir'],
                        '{}.csv'.format(name))
        if cache_file and os.path.exists(cache_file) and not refresh:
            st = os.stat(cache_file)
            now = datetime.now().timestamp()
            if now - st.st_mtime < t['cache_secs']:
                return pd.read_csv(cache_file)

        # fetch and cache
        self.providers[provider].throttle()

        url_params = {}
        for k in t['url_params']:
            url_params[k] = kwargs[k]
        url = t['url'].format(**url_params)

        data_params = {}
        for k in t['data_params']:
            data_params[k] = kwargs[k]
        # lookup api keys in our the key chain
        for k in t['key_map'].keys():
            v = t['key_map'][k]
            if v not in self.key_chain or self.key_chain[v] is None:
                raise ValueError("Required key '{}' is missing".format(v))
            data_params[k] = self.key_chain[v]

        data = None
        if t['data']:
            data = t['data'].format(data_params)

        if data:
            resp = requests.request(t['method'], url, json=data)
        else:
            resp = requests.request(t['method'], url, params=data_params)

        # pandify
        df = t['pandify_fn'](resp.text)

        # cache
        if cache_file and df is not None:
            df.to_csv(cache_file)

        return df
