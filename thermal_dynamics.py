__doc__ = """
Investigate the temperature relaxation of 1D MOT. 
"""


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import Akima1DInterpolator

import numpy as np
from tqdm import tqdm


class MOT1DModel:
    def __init__(self, s, delta, b):
        """ 
        Parameters: 
        s: normalized intensity 
        delta: normalized detuning 
        b: coefficient before x 
        """

        def prob1(x, u):
            tmp = u + b * x
            return s / (1 + s + 4 * (delta - tmp)**2) * 1.6  # counter

        def prob2(x, u):
            tmp = u + b * x
            return s / (1 + s + 4 * (delta + tmp)**2) * 1.6

        self.prob1 = prob1
        self.prob2 = prob2

    def simulate(self, u0, step, tot_steps=2000000):
        """
        In a thermodynamic setup, the atoms are assumed to located at the origin; 

        The SDE is solved with vanilla Euler method   
        """

        ys = [0]
        us = [u0]

        cnt = 0 
        for _ in tqdm(range(tot_steps)):
            delta_u = 0
            # scatter in 2D
            # # scatter a photon with -k
            # if np.random.rand() < self.prob1(ys[-1], us[-1]) * step:
            #     delta_u += +1/1.6 * (1 + np.cos(np.random.random()*np.pi))
            # # scatter a photon with +k
            # if np.random.rand() < self.prob2(ys[-1], us[-1]) * step:
            #     delta_u += -1/1.6 * (1 + np.cos(np.random.random()*np.pi))

            # pure 1D
            # scatter a photon with -k
            if np.random.rand() < self.prob1(ys[-1], us[-1]) * step /2 :
                delta_u += +1/1.6 
                cnt += 1
            # scatter a photon with +k
            if np.random.rand() < self.prob2(ys[-1], us[-1]) * step / 2:
                delta_u += -1/1.6 
                cnt += 1

            ys.append(ys[-1] + us[-1] * step)
            us.append(us[-1] + delta_u)

        print('Scattered rate:', cnt / tot_steps)
        return np.array(ys), np.array(us)


def binning(x, max_bin=400):
    def mean_every_n(arr, n):
        return np.mean(arr[:(len(arr)//n)*n].reshape(-1, n), axis=1)
    return np.array([np.std(mean_every_n(x, b)) for b in range(1, max_bin)])


model = MOT1DModel(284, -16, .048)
ys, us = model.simulate(0, .05)

# plt.plot(us)
# plt.show()


def autocorr(x, t=1):
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1]


burnin = 5000
ys = ys[burnin:]
us = us[burnin:]
print(us[:1000])
plt.hist2d(us, ys,bins=10)
# plt.plot(*np.unique(us, return_counts=True),'+')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.tight_layout()
plt.show()
lags = list(range(1, 5000, 20))
plt.plot(lags, [autocorr(us, l) for l in tqdm(lags)])
plt.show()

ys = ys[::1000]
us = us[::1000]

print(np.mean(us**2))
plt.plot(binning(us**2))
plt.show()
