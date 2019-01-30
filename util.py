import numpy as np
import torch
from torch.utils import data
from torch.nn.modules import Linear
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class F:
    def __init__(self, w, plot=True, range_x=[(0, 0), (0, 0)]):
        self.w = w
        self.range_x = range_x

        self.fn = lambda x1, x2: w[0]*(x1**2) + w[1]*(x2**2) + w[2]*(x1*x2) + w[3]*x1 + w[4]*x2 + w[5]
        self.grad_fn = lambda x1, x2: [w[0]*(2*x1) + w[2]*x2 + w[3], w[1]*(2*x2) + w[2]*x1 + w[4]]

        if plot:
            plot_contour_x(self, range_x)
            plt.show()

    def val(self, x):
        return np.array(self.fn(x[0], x[1]))

    def grad(self, x):
        return np.array(self.grad_fn(x[0], x[1]))


def plot_contour_x(fn, range_x):
    origin = 'lower'
    x1 = np.arange(range_x[0][0], range_x[0][1], (1.0 * range_x[0][1] - range_x[0][0]) / 100)
    x2 = np.arange(range_x[1][0], range_x[1][1], (1.0 * range_x[1][1] - range_x[1][0]) / 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = fn.val([X1, X2])

    Z = np.ma.array(Z)

    fig, ax = plt.subplots()
    CS = ax.contourf(X1, X2, Z, 10, cmap=plt.cm.bone, origin=origin)

    CS2 = ax.contour(CS, levels=CS.levels[::2], colors='r', origin=origin)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    return fig, ax