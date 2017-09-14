import numpy as np
import gzip
import sorting_with_python as swp

data_dir = 'C:/Users/dbahena/PycharmProjects/data-science-learning/machine-learning/training-oracle/expo-kmeans-pca-spike-sorting/'

# loads
def load_data():
    # Create a list with the file names
    data_files_names = [data_dir + 'data/Locust_' + str(i) + '.dat.gz' for i in range(1, 5)]

    # Load the (compressed) data in a list of numpy arrays
    load_file = lambda f: np.frombuffer(gzip.open(f).read())
    raw_data = map(load_file, data_files_names)

    # Get the length of the data in the files
    data_len = np.unique(list(map(len, raw_data)))[0]

    return raw_data, data_len
