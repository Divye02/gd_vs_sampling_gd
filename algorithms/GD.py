from algorithms.base_algo import BaseAlgo
import numpy as np

class GD(BaseAlgo):
    def __init__(self, f, lr, plot):
        super().__init__(f, lr, plot)

    def get_init(self):
        init_x = np.array([np.random.random_sample()*(self.range_x[0][1] - self.range_x[0][0]) + self.range_x[0][0], np.random.random_sample()*(self.range_x[1][1] - self.range_x[1][0]) + self.range_x[1][0]])
        print(init_x)
        return init_x

    def update_step(self, x):
            self.plot_i += 1
            return x - self.lr*self.f.grad(x)