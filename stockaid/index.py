
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


from html.parser import HTMLParser
from html import unescape
from .throttle import CrudeThrottler
import pandas as pd

class _WikiTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_textarea = False
        self.scraped = ""

    def handle_starttag(self, tag, attrs):
        if tag == 'textarea':
            for t in attrs:
                if t[0] == 'id' and t[1] == 'wpTextbox1':
                    self.in_textarea = True

    def handle_endtag(self, tag):
        if self.in_textarea and tag == 'textarea':
            self.in_textarea = False

    def handle_data(self, data):
        if self.in_textarea:
            self.scraped = self.scraped + data

    # unwrap link markup, e.g. [[key|text]]
    def _unwrap(self, s, d1, d2, d3):
        start = s.find(d1)
        if start >= 0:
            start = start + len(d1)
            end = s.find(d2, start)
            if end >= 0:
                s = s[start:end]
                start = s.find(d3)
                if start >= 0:
                    return [s[:start], s[start+1:]]
        return [s, ""]

    # extract columns into a list
    def _get_cols(self, row, delim):
        cols = []
        end = row.find(delim)
        while end >= 0:
            start = end + len(delim)
            end = row.find(delim, start)
            if end < 0:
                col = row[start:]
            else:
                col = row[start:end]

            # remove wiki markup
            # [[key]] or [[key|text]] is a link to a wiki page. Take text
            parts = self._unwrap(col, '[[', ']]', '|')
            if parts[1] == "":
                col = parts[0]
            else:
                col = parts[1]

            # [url text] is an external link. Take the url
            parts = self._unwrap(col, '[', ']', ' ')
            col = parts[0]

            # {{something|key}} is a wiki template. Take the key
            parts = self._unwrap(col, '{{', '}}', '|')
            if parts[1]:
                col = parts[1]
            else:
                col = parts[0]

            cols.append(col.strip())
        return cols

    # generator, yields one row at a time as a list of column values
    def wiki_rows(self):
        table = unescape(self.scraped)
        found_head = False
        start = table.find('wikitable')
        if start < 0:
            start = 0
        end = table.find('\n|-', start)
        while end > 0:
            start = end + 3
            end = table.find('\n|-', start)
            if end < 0:
                row = table[start:]
            else:
                row = table[start:end]

            # first row is column names, delimited by !!, rest by ||
            if not found_head:
                row = row.replace('\n', '!')
                col_names = self._get_cols(row, '!!')
                num_cols = len(col_names)
                found_head = True
                yield col_names
            else:
                row = row.replace('\n', '|')
                a = self._get_cols(row, '||')
                yield [a[0:num_cols]]


def pandify_index(html):
    """This pandify_fn converts a wiki table to a pandas DataFrame."""

    wtp = _WikiTableParser()
    wtp.feed(html)
    wtp.close()

    cols = None
    df = None
    for row in wtp.wiki_rows():
        if not cols:
            cols=row
        elif df is None:
            df = pd.DataFrame(row, columns=cols)
        else:
            r = pd.DataFrame(row, columns=cols)
            df = df.append(r)

    return df

# All of these are cached for a week. Throttled to 10 requests / min.
# index sp500      S&P 500 Index
# index OEX        S&P 100 Index
# index midcap     S&P MidCap 400 Index
# index smallcap   S&P SmallCap 600 Index
# index nasdaq100  Nasdaq 100
# index DJIA       Dow Jones Industrial Average
def register_index(cache):
    """This function registers the api call for the index provider. Normally,
       this is done for you as part of the stockaid.get_cache() function.
    """

    throt = CrudeThrottler(10)
    cache.register_provider('index', 'https://en.wikipedia.org/', throt)
    cache.register_api('index', 'sp500',
        "w/index.php?title=List_of_S%26P_500_companies&action=edit&section=1",
        None, pandify_index, cache_secs=604800)
    cache.register_api('index', 'OEX',
        'w/index.php?title=S%26P_100&action=edit&section=3',
        None, pandify_index, cache_secs=604800)
    cache.register_api('index', 'midcap',
        'w/index.php?title=List_of_S%26P_400_companies&action=edit&section=1',
        None, pandify_index, cache_secs=604800)
    cache.register_api('index', 'smallcap',
        'w/index.php?title=List_of_S%26P_600_companies&action=edit&section=1',
        None, pandify_index, cache_secs=604800)
    cache.register_api('index', 'nasdaq100',
        'w/index.php?title=Nasdaq-100&action=edit&section=13',
        None, pandify_index, cache_secs=604800)
    cache.register_api('index', 'DJIA',
        'w/index.php?title=Dow_Jones_Industrial_Average&action=edit&section=1',
        None, pandify_index, cache_secs=604800)
