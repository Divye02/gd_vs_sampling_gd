from algorithms.GD import GD
from util import F
def main():
    fn = F(w=[0.9, 0.8, -0.7, 0.3, 0.3, 1], plot=False, range_x=[(-1, 1), (-1, 1)])
    lr = 0.01
    gd = GD(fn, lr, plot=True)
    gd.evaluate(steps=1000)

if __name__ == '__main__':
    main()