from glob import glob
from os.path import join, basename

import pandas as pd
from matplotlib import pyplot as plt

from utils import new_fig, save_plot

DATA_DIR = 'data'
PLOTS_DIR = 'plots'

def plot_lines(data_file, plot_file):
    data = pd.read_csv(data_file)
    fig = new_fig()
    xlab = 'Use time of Head'
    ylab = 'Temp. THG'
    for hd in data['S/N of Head'].unique():
        print('Generating lines for head %s' % hd)
        data_hd = data.loc[data['S/N of Head'] == hd]
        x = data_hd[xlab]
        y = data_hd[ylab]
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.plot(x, y, label=hd)
        plt.scatter(x, y)
    plt.legend(loc='upper right')
    save_plot(plot_file, fig)


if __name__ == '__main__':
    for data_file in glob(join(DATA_DIR, 'toollog*.csv')):
        plot_file = basename(data_file).replace('.csv', '.png')
        plot_file = join(PLOTS_DIR, plot_file)
        plot_lines(data_file, plot_file)


