from algorithms.base_algo import BaseAlgo
import numpy as np


class SamplingGD(BaseAlgo):
    def __init__(self, f, lr, plot, num_samples=10, resample_step=10, kappa=0.1, epsilon=0.9):
        super().__init__(f, lr, plot)
        self.num_samples = num_samples
        self.resample_step = resample_step
        self.kappa = kappa
        self.epsilon = epsilon

    def get_init(self, x_dim=2):
        self.update_count = 0
        self.mean = np.random.random(x_dim)/2
        cov = np.random.random((x_dim, x_dim))/2
        self.cov = np.dot(cov, np.transpose(cov))

        return np.random.multivariate_normal(self.mean, self.cov, self.num_samples)

    def update_step(self, x):
        x = x - self.lr * self.f.grad(x)
        if (self.update_count + 1) % 10 == 0:
            print(self.mean)
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
            return np.random.multivariate_normal(self.mean, self.cov, self.num_samples)
        self.update_count += 1
        return x
