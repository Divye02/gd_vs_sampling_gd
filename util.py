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


class SimpleDataset(data.Dataset):
    def __init__(self, in_out,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.norm = LA.norm(in_out, ord=np.inf, axis=0)
        #
        np.random.shuffle(in_out)
        self.train_set = in_out[1000:2000, :];
        self.train_in = np.concatenate((
            self.train_set[:, 0:1] * self.train_set[:, 0:1], self.train_set[:, 1:2] * self.train_set[:, 1:2],
            self.train_set[:, 0:1] * self.train_set[:, 1:2], self.train_set[:, 0:1], self.train_set[:, 1:2]),
            axis=1)

        self.train_mean = np.mean(self.train_in, axis=0)
        self.train_std = np.sqrt(np.var(self.train_in, axis=0))

        self.out_mean = np.mean(self.train_in[:, 2])
        self.out_std = np.sqrt(np.var(self.train_in[:, 2]))

        # self.val_set = (in_out[0:1000, :]- self.train_mean)/ self.train_std
        self.val_set = in_out[0:1000, :]
        val_in = np.concatenate((
                       self.val_set[:, 0:1] * self.val_set[:, 0:1], self.val_set[:, 1:2] * self.val_set[:, 1:2],
                       self.val_set[:, 0:1] * self.val_set[:, 1:2], self.val_set[:, 0:1], self.val_set[:, 1:2]),
                       axis=1)

        val_in = (val_in - self.train_mean)/self.train_std
        val_out = (self.val_set[:, 2:3] - self.out_mean)/self.out_std
        self.val_dict = {'in': torch.from_numpy(val_in).float(), 'out': torch.from_numpy(val_out).float()}


        self.in_out = in_out
        self.transform = transform

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, idx):
        res_in = (self.train_in[idx, :] - self.train_mean)/ self.train_std
        res_out = (self.train_set[idx, 2:3] - self.out_mean)/self.out_std
        # # res = self.train_in[idx, :]
        # inp = np.concatenate(([res[0] * res[0]], [res[1] * res[1]], [res[0] * res[1]], [res[0]], [res[1]]))
        sample = {'in': torch.from_numpy(res_in).float(), 'out': torch.from_numpy(res_out).float()}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class ModifiedLinear(Linear):
    def __init__(self, in_features, out_features, weigth: np.ndarray, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(weigth)

    def reset_parameters(self, weigth: np.ndarray):
        self.weight = Parameter(torch.from_numpy(weigth).float())
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

