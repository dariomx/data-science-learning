"""
Miscellaneous utilities that do not fit anywhere else
"""

import threading

from matplotlib import pyplot as plt, gridspec


class MTCounter:
    """
    Multi-threaded counter
    """
    def __init__(self, val=0):
        self.val = val
        self.lock = threading.Lock()

    def incr(self, val=1):
        with self.lock:
            self.val += val

    def get(self):
        with self.lock:
            return self.val


_fig_cnt = MTCounter()


def new_fig(title=None, nrows=1, ncols=1):
    """Generates a new matplotlib"""
    _fig_cnt.incr()
    fig = plt.figure(num=_fig_cnt.get())
    if title:
        fig.suptitle(title)
    gridspec.GridSpec(nrows, ncols)
    return fig

def save_plot(plot_file, fig):
    """
    Saves the given figure into the plot_file

    TODO: allow resolution parameters to be passed?
    """
    dpi = 200
    figsize = (400 / dpi, 1200 / dpi)
    fig.savefig(plot_file, figsize=figsize, dpi=dpi)
    plt.close(fig)
