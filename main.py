from algorithms.GD import GD
from algorithms.SamplingGD import SamplingGD
from util import F, F2, F4
def main():
    # fn = F2(w=[0.9, 0.8, -0.7, 0.3, 0.3, 1], plot=False, range_x=[(-1, 1), (-1, 1)])
    fn = F4(w=[0.5, 0.5, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0], plot=False, range_x=[(-2, 2), (-2, 2)])

    # fn = fn_4
    lr = 0.002
    gd = GD(fn, lr, plot=True)
    x_gd = gd.evaluate(steps=1000)
    sgd = SamplingGD(fn, lr, plot=True, resample_step=10, epsilon=0.5, kappa=0.01)
    x_sgd = sgd.evaluate(steps=100)

    print('Value of f through GD: %.2f' % fn.val(x_gd))
    print(x_gd)
    print('Value of f through Sampling GD: %.2f' % fn.val(x_sgd))
    print(x_sgd)

if __name__ == '__main__':
    main()