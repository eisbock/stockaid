
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

"""MiniZinc is a constraint solver that compiles a high level language into
   an intermediate language that is understood by a large number of
   optimized solvers, although it also provides its own, called FlatZinc.
   A MiniZinc program will read a .dzn data file, and FlatZinc will output a
   file. You will likely want to convert between pandas DataFrames and
   these files, and special functions are provided to do this.
"""

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from shutil import copyfile
from subprocess import run
import pandas as pd
from ..logging import log

def dzn(df, coltype):
    """Transforms a pandas DataFrame into a string formated for a .dzn file.
       MiniZinc has strict typing, so a dict, coltype, maps column names to a
       type. MiniZinc functions best when its variables are int, float, or
       bool. There is good support for enumerated values, where a provided
       string is mapped to an int. Enums have names, so provide one as the
       type name. We will make sure that the enum type is defined before
       the data. You can specify a type of string, but MiniZinc has limited
       support for strings.

       :param df: the pandas DataFrame that contains the data
       :param col_type: a map between column name and type. Types can be
	   bool, int, float, string, or the name of an enum type
       :returns: a string
    """
    if not coltype or df is None:
        return None
    out=""

    # persuade Series or ndarray to become a comma separated string
    def comma_list(s):
        return ','.join(str(x) for x in s)

    # emumerated type definitions must go first
    enum = {}
    # loop once to make sure we have all the value
    for k in coltype:
        t = coltype[k]
        if t not in ['int','float','bool','string','len']:
            if enum.get(t) is None:
                enum[t] = df[k]
            else:
                enum[t] = enum[t].append(df[k])
    # then loop through the enums and output a unique set of values
    for t in enum:
        s = enum[t].unique()
        out = out + "{} = {{ {} }};\n".format(t, comma_list(s))

    # this is clearer than a lambda. Allow any type with a truth value.
    def convert_bool(x):
        if x:
            return 'true'
        else:
            return 'false'

    # iterate through columns and output
    for k in coltype:
        t = coltype[k]

        # variables of type len are not actually in the DataFrame
        if t == 'len':
            out = out + "{} = {};\n".format(k, len(df))
            continue

        # format the column data
        s = df[k]
        if t == 'bool':
            s = s.apply(convert_bool) # MiniZinc bools are lower case
        elif t == 'int':
            s = s.astype(int)
        elif t == 'float':
            s = s.astype(float)
        elif t == 'string':
            s = s.apply(lambda x: '"' + x + '"') # strings need to be quoted
        out = out + "{} = [{}];\n".format(k, comma_list(s))

    return out


def spawn(mzn_path, df, coltype, multiple=False, solver=None):
    """Export data to a .dzn file, run the minizinc program in mzn_path,
       and import the FlatZinc output file as a one or more Result instances.

       :param mzn_path: the path to the MiniZinc program.
       :param df: the pandas DataFrame that contains the data
       :param col_type: a map between column name and type. Types can be
	   bool, int, float, string, or the name of an enum type
       :param multiple: if True, the mzn program can return multiple reults.
       :param solver: if provided, this is passed to Minizinc --solver
       :returns: either a Result, if multiple is False, or an array of Result.
    """
    dzn_str = dzn(df, coltype)
    res_str = None

    # sanatize mzn_path, solver to discourage string injection
    dirty = ['$','..','&','<','>','!','?','~','(',')','{','}','[',']',';','#',
             '`','\\','=','|','*',' ','\t','\r','\n']
    if mzn_path and any(x in mzn_path for x in dirty):
        log(1, 'Possible string injection in MiniZinc.spawn() mzn_path')
        raise ValueError('solver is rejected to avoid string injection')
    if solver and any(x in solver for x in dirty):
        log(1, 'Possible string injection in Minizinc.spawn() solver')
        raise ValueError('solver is rejected to avoid string injection')

    # MiniZinc makes a mess, so isolate it in a temp dir
    with TemporaryDirectory() as temp_dir:
        mzn_temp = os.path.join(temp_dir, 'frompy.mzn')
        dzn_temp = os.path.join(temp_dir, 'frompy.dzn')

        fz_args = ['flatzinc', '-o', 'fz.out']
        if multiple:
            fz_args.append('-a')
        if solver:
            fz_args.append('--solver')
            fz_args.append(solver)
        fz_args.append('frompy.fzn')

        # copy in the program, write the data file
        copyfile(mzn_path, mzn_temp)
        with open(dzn_temp, 'w') as f:
            f.write(dzn_str)

        # compile MiniZinc into a FlatZinc program
        start = datetime.now().timestamp()
        done = run(['minizinc', '-c', mzn_temp, dzn_temp], cwd=temp_dir)
        end = datetime.now().timestamp()
        if done.returncode != 0:
            log(1, 'MiniZinc for {} returned error code {}.'.format(mzn_path,
                done.returncode))
            return None
        log(2, 'Compiled {} in {:.3f} seconds'.format(mzn_path, end-start))

        # use solver to solve FlatZinc program
        start = end
        done = run(fz_args, cwd=temp_dir)
        end = datetime.now().timestamp()
        if done.returncode != 0:
            log(1, 'Flatzinc for {} returned error code {}.'.format(mzn_path,
                done.returncode))
            return None
        log(2, 'FlatZinc ran in {:.3f} seconds.'.format(end-start))

        # read the output file
        with open(os.path.join(temp_dir, 'fz.out'), 'rt') as f:
            res_str = f.read()

    # iterate through res_str creating Result objects
    ret = []
    while res_str and res_str[:3] != "===":
        result = Result(res_str)
        if result.str_len == -1:
            res_str = None
        else:
            res_str = res_str[result.str_len:]
        if multiple:
            ret.append(result)
        else:
            return result
    if multiple:
        return ret
    return None


class Result():
    """This class reads strings that are output by FlatZinc."""

    str_len = 0
    data = {}

    def __init__(self, res_str):
        """Import into data dict the values found in s. The format of s should
           match the format of either a .dzn or FlatZinc output

           :param res_str: a string containing FlatZinc output
        """
        while res_str and res_str[self.str_len:self.str_len+3] != '---':
            idx = res_str.find('\n', self.str_len)
            if idx < 0:
                self.str_len = -1
                break
            else:
                self._parse_line(res_str[self.str_len:idx])
                self.str_len = idx + 1


    def get_df(self, columns):
        """This will extract the data values specified in the columns list into
           a pandas DataFrame. Each MiniZinc array variable with a name in
           columns should be of the same length.

           :param columns: an array of keys to this dict to use as columns
               in the returned DataFrame.
           :returns: a pandas DataFrame.
        """
        if not columns:
            return None
        s = {}
        for c in columns:
            s[c] = pd.Series(self.data[c])
        return pd.concat(s, axis=1)


    def _unquote(self, val):
        if val[0:1] == '"':
            return val[1:len(val)-1]
        elif val.find('[') >= 0:
            start = val.find('[') + 1
            end = val.find(']', start)
            return self._parse_array(val[start:end])
        elif val.find('.') >= 0:
            return float(val)
        elif val.isnumeric():
            return int(val)
        elif val == 'false':
            return False
        elif val == 'true':
            return True
        else:
            return val

    def _parse_array(self, arr):
        a = []
        start = 0
        while True:
            idx = arr.find(',', start)
            if idx < 0:
                a.append(self._unquote(arr[start:].strip()))
                return a
            else:
                a.append(self._unquote(arr[start:idx].strip()))
                start = idx + 1

    def _parse_line(self, line):
        # label = value;
        idx = line.find('=')
        if (idx < 0):
            return
        label = line[0:idx].strip()
        value = line[idx+1:].strip()
        idx = value.find(';')
        value = value[0:idx].strip()
        value = self._unquote(value)
        if label is not None and value is not None:
            self.data[label] = value

