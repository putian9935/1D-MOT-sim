import matplotlib.pyplot as plt
__doc__ = "Repeat what is done in sec 5.1"

import enum
import numpy as np
from numpy.lib import unique
from cg_coefficient import cg
from tqdm import tqdm


class tSimulator:
    def __init__(self, s, delta, gamma=1.6, nmax=100):
        self.max_momentum = nmax
        self.alpha = np.zeros(2 * self.max_momentum + 1,
                              dtype=np.complex128)  # excited state
        self.beta = np.zeros(2 * self.max_momentum + 1,
                             dtype=np.complex128)  # ground state
        self.beta[-self.max_momentum//2] = 1
        self.s = s
        self.delta = delta
        self.gamma = gamma

        self.p_bar = self.gamma * np.array(
            [1/5, 3/5, 1/5],
        )

        self.p_mat = np.diag(
            list(range(-self.max_momentum, self.max_momentum+1)))

    def jump(self, time_step):
        """Determine whether a jump is made and, if yes, and where it is to. 

        Returns
        -------
        state: a number of 0 to 3
            for state 0 - 2, the corresponding jump is made; 
            for state 3, no jump is made 
        """

        distribution = np.hstack(
            (0, time_step * self.p_bar * np.sum(np.abs(self.alpha))))
        distribution.cumsum(out=distribution)
        return np.searchsorted(distribution, np.random.random()) - 1

    def simulate(self, tot_steps, time_step, stat_funcs):
        ret = []
        for _ in tqdm(range(tot_steps)):
            jump = self.jump(time_step)
            if jump == 3:  # no jump
                new_alpha = np.array(self.alpha, dtype=np.complex128)
                new_beta = np.array(self.beta, dtype=np.complex128)
                for ip, p in enumerate(range(-self.max_momentum, self.max_momentum+1)):
                    new_alpha[ip] += time_step * \
                        (-1j * p*p+self.gamma *
                         (1j*self.delta-.5))*self.alpha[ip]
                    new_beta[ip] += time_step * -1j * p*p * self.beta[ip]
                    if p + 1 <= self.max_momentum:
                        new_alpha[ip] += -.5j * time_step * \
                            self.s ** .5 * self.beta[ip+1]
                        new_beta[ip] += -.5j * time_step * \
                            self.s ** .5 * self.alpha[ip+1]
                    if p - 1 >= -self.max_momentum:
                        new_alpha[ip] += -.5j * time_step * \
                            self.s ** .5 * self.beta[ip-1]
                        new_beta[ip] += -.5j * time_step * \
                            self.s ** .5 * self.alpha[ip-1]

                self.alpha = np.array(new_alpha, dtype=np.complex128)
                self.beta = np.array(new_beta, dtype=np.complex128)

            else:
                if jump == 2:
                    self.beta[:-1] = self.alpha[1:]
                    self.beta[-1] = 0
                elif jump == 1:
                    self.beta = self.alpha
                else:
                    self.beta[1:] = self.alpha[:-1]
                    self.beta[0] = 0

                self.alpha = np.zeros_like(self.alpha, dtype=np.complex128)

            tot_mod = np.sum(np.abs(self.alpha)**2) + \
                np.sum(np.abs(self.beta)**2)
            self.alpha /= tot_mod ** .5
            self.beta /= tot_mod ** .5

            new_entry = [jump]
            for func in stat_funcs:
                new_entry.append(func(self))

            ret.append(new_entry)
        return np.array(ret)

    def momentum(self):
        return (np.conjugate(self.alpha).T @ self.p_mat @ self.alpha) + \
            (np.conjugate(self.beta).T @ self.p_mat @ self.beta)

    def ground_state_prob(self):
        return (np.conjugate(self.beta).T  @ self.beta)


result = tSimulator(
    1, -.1, nmax=8,).simulate(10000, .001, [tSimulator.momentum, tSimulator.ground_state_prob])
print(*np.unique(result[:, 0].real, return_counts=True))
fig, ax = plt.subplots()
ax.plot(result[:, 1].real)
ax2 = ax.twinx()
ax2.plot(result[:, 2].real, c='C1')
plt.show()
