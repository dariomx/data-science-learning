import numpy as np
from load_data import load_data
import sorting_with_python as swp

data, data_len = load_data()
tt = np.arange(0, data_len) / 1.5e4
swp.plot_data_list(data, tt, 0.1)
