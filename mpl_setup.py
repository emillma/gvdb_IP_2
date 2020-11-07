import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy as np
from numpy.core.shape_base import block

mpl.rcParams["savefig.directory"] = os.getcwd()


def resize_event(event):
    print('hello')


plt.close('all')
fig, ax = plt.subplots(1)
ax.plot(np.arange(10))
fig.canvas.mpl_connect('resize_event', resize_event)
# plt.show(block=True)
