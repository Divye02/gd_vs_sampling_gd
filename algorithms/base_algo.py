from util import plot_contour_x
import matplotlib.pyplot as plt

class BaseAlgo:

    def __init__(self, f, lr, plot=True):
        self.f = f
        self.plot = plot
        self.lr = lr
        self.range_x = f.range_x
        self.dx = [(1.0*self.range_x[0][1] - self.range_x[0][0])/100, (1.0*self.range_x[1][1] - self.range_x[1][0])/100]
    def get_init(self):
        pass

    def update_step(self, x):
        pass

    def evaluate(self, steps=1000):
        fig, ax = plot_contour_x(self.f, self.range_x)
        x = self.get_init()

        ax.plot(x[0], x[1], 'bo')
        ax.annotate('(%.2f, %.2f)'% (x[0], x[1]), (x[0], x[1]), (x[0]+self.dx[0], x[1]+self.dx[1]))

        for i in range(0, steps):
            x = self.update_step(x)
            ax.plot(x[0], x[1], 'bo')
            # ax.annotate(str(i+1), (x[0], x[1]), (x[0] + self.dx[0], x[1] + self.dx[1]))

        if self.plot:
            plt.show()

        return x



