
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

_stockaid_user_logging_fn = None
_stockaid_log_level = 2

def set_log_fn(fn):
    """Define an external function to use for logging messages. This function
       should except a single string as an argument.
    """
    global _stockaid_user_logging_fn
    _stockaid_user_logging_fn = fn


def set_log_level(l):
    """Define the log level required for messages to be printed or sent to the
       log function. Values can be:
           3 debug messages
           2 (default) info or warnings
           1 errors
           0 no messages
    """
    global _stockaid_log_level
    _stockaid_log_level = l


def log(level, s):
    global _stockaid_user_logging_fn
    global _stockaid_log_level

    if level > _stockaid_log_level:
        return

    if _stockaid_user_logging_fn is None:
        print(s)
    else:
        _stockaid_user_logging_fn(s)

