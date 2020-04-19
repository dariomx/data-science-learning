from datetime import datetime as dt
from os.path import dirname, realpath, join, basename
from time import process_time

import pandas as pd


def get_fullpath(path):
    bdir = dirname(realpath(__file__))
    return join(bdir, path)


def logmsg(fmt, *args):
    ts = dt.now().strftime('%H:%M:%S')
    print(ts + ': ' + (fmt % args))


def get_data(data_file, cols=None):
    logmsg('About to load data from %s ...', data_file)
    start = process_time()
    if cols is None:
        data = pd.read_csv(data_file)
    else:
        data = pd.read_csv(data_file, usecols=cols)
    end = process_time()
    logmsg('Reading %s data took %f secs', basename(data_file), end - start)
    return data
