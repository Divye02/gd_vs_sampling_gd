import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


class F:
    def __init__(self, w, range_x=[(0, 0), (0, 0)]):
        self.w = w
        self.range_x = range_x

    def val(self, x):
        pass

    def grad(self, x):
        pass


class F2(F):
    def __init__(self, w, plot=True, range_x=[(0, 0), (0, 0)]):
        super().__init__(w, range_x)

        self.fn = lambda x1, x2: w[0]*(x1**2) + w[1]*(x2**2) + w[2]*(x1*x2) + w[3]*x1 + w[4]*x2 + w[5]
        self.grad_fn = lambda x1, x2: [w[0]*(2*x1) + w[2]*x2 + w[3], w[1]*(2*x2) + w[2]*x1 + w[4]]

        if plot:
            plot_contour_x(self, range_x)
            plt.show()

    def val(self, x):
        return np.array(self.fn(x[0], x[1]))

    def grad(self, x):
        if len(x.shape) > 1:
            return np.array([self.grad_fn(x_i[0], x_i[1]) for x_i in x])
        return np.array(self.grad_fn(x[0], x[1]))

# https://www.wolframalpha.com/input/?i=plot+.5x%5E4+-+x%5E2+%2B.5y%5E4++%2B+0.6x%5E3
class F4(F):
    def __init__(self, w, plot=True, range_x=[(0, 0), (0, 0)]):
        super().__init__(w, range_x)

        self.fn = lambda x1, x2: w[0]*(x1 ** 4) + w[1]*(x2**4) + w[2]*(x1**3 * x2) + w[3]*(x1 * x2**3) + w[4]*(x1**2 * x2**2) \
                                 + w[5]*(x1 ** 3) + w[6]*(x2 ** 3) + w[7]*(x1**2 * x2) + w[8]*(x1 * x2**2) \
                                 + w[9]*(x1 ** 2) + w[10]*(x2 ** 2) + w[11]*(x1 * x2) \
                                 + w[12]*x1 + w[13]*x2 \
                                 + w[14]
        self.grad_fn = lambda x1, x2: [w[0]*(4* x1 ** 3) + w[2]*(3 * x1**2 * x2) + w[3]*(x2**3) + w[4]*(2*x1 * x2**2) \
                                 + w[5]*(3*x1 ** 2) + w[7]*(2*x1 * x2) + w[8]*(x2**2) \
                                 + w[9]*(2*x1) + w[11]*(x2) \
                                 + w[12],
                                   w[1] * (4*x2 ** 3) + w[2] * (x1 ** 3) + w[3] * (3*x1 * x2 ** 2) + w[4] * (2*x1 ** 2 * x2) \
                                 + w[6] * (3*x2 ** 2) + w[7] * (x1 ** 2) + w[8] * (2*x1 * x2) \
                                 + w[10] * (2*x2) + w[11]*(x1) \
                                 + w[13]]

        if plot:
            plot_contour_x(self, range_x)
            plt.show()

    def val(self, x):
        return np.array(self.fn(x[0], x[1]))

    def grad(self, x):
        if len(x.shape) > 1:
            return np.array([self.grad_fn(x_i[0], x_i[1]) for x_i in x])
        return np.array(self.grad_fn(x[0], x[1]))

def plot_contour_x(fn, range_x):
    origin = 'lower'
    x1 = np.arange(range_x[0][0], range_x[0][1], (1.0 * range_x[0][1] - range_x[0][0]) / 1000)
    x2 = np.arange(range_x[1][0], range_x[1][1], (1.0 * range_x[1][1] - range_x[1][0]) / 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = fn.val([X1, X2])

    Z = np.ma.array(Z)

    fig, ax = plt.subplots()
    CS = ax.contourf(X1, X2, Z, 20, cmap=plt.cm.bone, origin=origin)

    CS2 = ax.contour(CS, levels=CS.levels[::2], colors='r', origin=origin)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    return fig, ax

def plot_gaussian_contour(mean, cov, color):
    X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    xy = np.column_stack([X.flat, Y.flat])

    # density values at the grid points
    Z = mvn.pdf(xy, mean, cov).reshape(X.shape)

    # arbitrary contour levels
    contour_level = [0.1, 0.2, 0.3]

    fig = plt.contour(X, Y, Z, levels=4, colors=color)