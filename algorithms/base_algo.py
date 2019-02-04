from util import plot_contour_x
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
class BaseAlgo:

    def __init__(self, f, lr, plot=True):
        self.f = f
        self.plot = plot
        self.lr = lr
        self.range_x = f.range_x
        self.dx = [(1.0*self.range_x[0][1] - self.range_x[0][0])/100, (1.0*self.range_x[1][1] - self.range_x[1][0])/100]
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Sort colors by hue, saturation, value and name.
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                        for name, color in colors.items())
        self.sorted_names = np.array([name for hsv, name in by_hsv])
        indexes = np.arange(len(self.sorted_names))
        np.random.shuffle(indexes)
        self.sorted_names = self.sorted_names[indexes]
        self.plot_i = 0
    def get_init(self):
        pass

    def update_step(self, x):
        pass

    def ret_val(self, x):
        return x

    def evaluate(self, steps=1000):
        fig, ax = plot_contour_x(self.f, self.range_x)
        x, plot_x = self.get_init()
        x_dim = 2
        plot_x = plot_x.reshape(x_dim, -1)
        ax.plot(plot_x[0], plot_x[1], 'bo', color=self.sorted_names[self.plot_i % len(self.sorted_names)])
        # ax.annotate('', (x_p[0], x_p[1]), (x_p[0]+self.dx[0], x_p[1]+self.dx[1]))

        for i in range(0, steps):
            x, plot_x = self.update_step(x)
            plot_x = plot_x.reshape(x_dim, -1)
            # print(self.plot_i)
            ax.plot(plot_x[0], plot_x[1], 'bo', color=self.sorted_names[self.plot_i % len(self.sorted_names)])
            # ax.annotate(str(i+1), (x[0], x[1]), (x[0] + self.dx[0], x[1] + self.dx[1]))

        if self.plot:
            plt.show()

        return self.ret_val(x)



