import numpy as np
from load_data import load_data
import sorting_with_python as swp

raw_data, _, data_len = load_data()
tt = np.arange(0, data_len) / 1.5e4
swp.plot_data_list(raw_data, tt, 0.1)
