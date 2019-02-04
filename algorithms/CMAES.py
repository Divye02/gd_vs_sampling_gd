from algorithms.base_algo import BaseAlgo
import numpy as np
from util import plot_gaussian_contour

class CMAES(BaseAlgo):
    def __init__(self, f, lr, plot, num_samples=10, kappa=0.1, epsilon=0.9):
        super().__init__(f, lr, plot)
        self.num_samples = num_samples
        self.kappa = kappa
        self.epsilon = epsilon

    def get_init(self, x_dim=2, range_x=[[-1, 1], [-1, 1]]):
        self.update_count = 0
        # self.mean = np.random.random(x_dim) * [range_x[0][1] - range_x[0][0], range_x[1][1] - range_x[1][0]] + [range_x[0][0], range_x[1][0]]
        self.mean = np.array([1, 1])
        cov = np.zeros((x_dim, x_dim))
        rand_n1 = np.random.random()
        rand_n2 = np.random.random()
        cov[0][0] = rand_n1
        cov[1][1] = rand_n1
        cov[0][1] = rand_n2
        cov[1][0] = rand_n2
        # self.cov = np.dot(cov, np.transpose(cov))
        self.cov = cov
        self.cov = np.dot(cov, np.transpose(cov))
        print("CMA-ES", self.mean)
        print("CMA-ES", self.cov)
        # plot_gaussian_contour(self.mean, self.cov, self.sorted_names[self.plot_i % len(self.sorted_names)])
        x_samples = np.clip(np.random.multivariate_normal(self.mean, self.cov, self.num_samples), range_x[0][0], range_x[0][1])
        return x_samples, self.mean

    def ret_val(self, x):
        return self.mean

    def update_step(self, x):
        # print(self.mean)
        self.plot_i += 1
        # calculate weights of the samples
        weight_samples = np.array([np.exp(-(1.0 / self.kappa) * self.f.val(x_i)) for x_i in x])
        weight_samples = weight_samples / np.sum(weight_samples)

        # recalculate mean
        self.mean = np.sum(np.multiply(weight_samples.reshape(-1, 1), x), axis=0)

        # recalculate cov
        self.cov = self.epsilon * np.sum(
            np.array([weight_samples[i] * (x_i - self.mean).reshape(-1, 1) @ (x_i - self.mean).reshape(1, -1) for i, x_i in enumerate(x)]), axis=0) + (
                           1 - self.epsilon) * self.cov
        # sample x
        # plot_gaussian_contour(self.mean, self.cov, self.sorted_names[self.plot_i % len(self.sorted_names)])
        x_new = np.random.multivariate_normal(self.mean, self.cov, self.num_samples)
        self.update_count += 1
        return x_new, self.mean
