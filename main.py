from algorithms.GD import GD
from algorithms.SamplingGD import SamplingGD
from algorithms.CMAES import CMAES
from util import F, F2, F4
import numpy as np
def main(seed=100):
    np.random.seed(seed)
    # fn = F2(w=[0.9, 0.8, -0.7, 0.3, 0.3, 1], plot=False, range_x=[(-1, 1), (-1, 1)])
    fn = F4(w=[0.3, 0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0], plot=False, range_x=[(-2, 2), (-2, 2)])

    # fn = fn_4
    lr = 0.002
    gd = GD(fn, lr, plot=True)
    x_gd = gd.evaluate(steps=1000)
    print('Value of f through GD: %.2f' % fn.val(x_gd))
    print(x_gd)
    print()

    cmaes = CMAES(fn, lr, plot=True, epsilon=0.5, kappa=0.01)
    x_cmaes = cmaes.evaluate(steps=1000)
    print('Value of f through CMA-ES: %.2f' % fn.val(x_cmaes))
    print(x_cmaes)
    print()

    sgd = SamplingGD(fn, lr, plot=True, resample_step=100, epsilon=0.5, kappa=0.01)
    x_sgd = sgd.evaluate(steps=1000)
    print('Value of f through Sampling GD: %.2f' % fn.val(x_sgd))
    print(x_sgd)
    print()


if __name__ == '__main__':
    # Work 4/5 times
    main(0)
    main(100)
    main(200)
    main(5000)
    main(20)
