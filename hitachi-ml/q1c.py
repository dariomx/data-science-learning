from glob import glob
from os.path import join

import pandas as pd

DATA_DIR = 'data'
DATAGEN_DIR = 'datagen'

# we materialize all data in memory; if that is a problem due space
# considerations, we could rather be appending to the file once in a while.
# assumes all files have same layout; otherwise panda api will complain
def combine_files(out_data_file, *in_data_files):
    if len(in_data_files) == 0:
        raise ValueError('At least one data file must be passed')
    print('Appending data file %s ... ' % in_data_files[0])
    data = pd.read_csv(in_data_files[0])
    for data_file in in_data_files[1:]:
        print('Appending data file %s ... ' % data_file)
        data = data.append(pd.read_csv(data_file))
    print('Saving all data into %s ...' % out_data_file)
    data.to_csv(out_data_file, index=False)


# sample usage of the function
if __name__ == '__main__':
    in_data_files = glob(join(DATA_DIR, 'toollog*.csv'))
    out_data_file = join(DATAGEN_DIR, 'all-toollog.csv')
    combine_files(out_data_file, *in_data_files)
