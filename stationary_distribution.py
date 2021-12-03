__doc__ = """Solve for stationary momentum distribution of J=0 <-> J=1 transition. 

The full master equation will be solved (in a subspace of diagonal momentums) and the results should match. 
"""

import numpy as np
from functools import lru_cache
from tqdm import tqdm

import matplotlib.pyplot as plt


class Solver:
    r"""
    Matrix layout: 
    9 density matrix elements \times momentum_states
    0 : \pi_g
    1 : \pi_+
    2 : \pi_- 
    3 : Re\rho_{g+}
    4 : Im\rho_{g+}
    5 : Re\rho_{g-}
    6 : Im\rho_{g-}
    7 : Re\rho_{+-}
    8 : Im\rho_{+-}
    """

    def __init__(self, s, delta, nmax=(20, 10), gamma=1.6, r=16):
        self.s = s
        self.delta = delta
        self.gamma = gamma
        self.r = r
        self.max_momentum = nmax[0]
        self.momentum_per_recoil = nmax[1]
        self.momentum_states = (2*self.max_momentum) * \
            self.momentum_per_recoil

        self.momentums = np.arange(-self.momentum_per_recoil*self.max_momentum,
                                   self.momentum_per_recoil*self.max_momentum) / self.momentum_per_recoil

        self.n_u = self._calc_n_u()
        self.simpson_coeff = self._calc_simpson_coeff(len(self.n_u))
        self.n_u = self.simpson_coeff * self.n_u
        self.m = self._calc_mat()

    def _calc_n_u(self):
        """Pre-calculate N(u) in the range of [-1,1]"""
        u = np.arange(-self.momentum_per_recoil,
                      self.momentum_per_recoil + 1) / self.momentum_per_recoil
        return 3/8 * (1 + u ** 2)

    @staticmethod
    @lru_cache(maxsize=None)
    def _calc_simpson_coeff(length):
        """The Simpson 1/3 rule. 

        Parameter: 
        ----------
        - length : the number of sampled points
        """

        if length == 2:
            return (.5, .5)
        if length == 3:
            return (1/3, 4/3, 1/3)
        if length == 4:
            return (3/8, 9/8, 9/8, 3/8)

        if length % 2:
            return tuple([1/3] + [4/3, 2/3]*((length-3)//2) + [4/3, 1/3])
        return tuple([1/3] + [4/3, 2/3]*((length-6)//2) + [4/3, 1/3+3/8, 9/8, 9/8, 3/8])

    @staticmethod
    def _calc_trapz_coeff(length):
        """The trapzoid rule. 

        Parameter: 
        ----------
        - length : the number of sampled points
        """

        return ([.5]+[1]*(length-2)+[.5])

    def _calc_mat(self):
        r"""
        Calculate the matrix of evolution. 

        The matrix is split into 4 parts: 

        1. Those involving \bar\pi thus integral  
        2. Those related with d/du derivative due to gravity 
        3. The rest (u dependent)
        4. The rest (u independent)

        This way, each part has a clear structure.  

        The structure of the full matrix is similar to the result np.kron(operators, momentum), i.e.
        it can be regarded as a 9x9 block matrix. 
        """

        def _kron(a, b):
            return (a[:, None, :, None]*b[None, :, None, :]).reshape(9*self.momentum_states, 9*self.momentum_states)

        part_ia = np.zeros((self.momentum_states, self.momentum_states))
        part_ib = np.zeros((self.momentum_states, self.momentum_states))

        tabbed = np.hstack(
            [self.n_u, [0] * (self.momentum_states - len(self.n_u))])
        for idx in range(len(self.momentums)):
            part_ia[idx, :] = np.roll(tabbed, -len(self.n_u)+idx+1)
            part_ib[idx, :] = np.roll(tabbed, idx)

        buf = np.zeros((9, 9))

        buf[0, 1] = 1.
        part_i = _kron(part_ia, buf)
        buf[0, 1] = 0.

        buf[0, 2] = 1.
        part_i += _kron(part_ib, buf)

        part_i /= self.momentum_per_recoil

        #####

        part_ii = np.diag([1]*(self.momentum_states-1), 1) - \
            np.diag([1]*(self.momentum_states-1), -1)
        part_ii[0, -1] = -1
        part_ii[-1, 0] = 1

        part_ii = _kron(part_ii, np.eye(9)) / \
            (2. * self.r) * self.momentum_per_recoil

        #####

        part_iii = np.zeros((9, 9))
        part_iii[3, 4] = part_iii[6, 5] = - 2.
        part_iii[4, 3] = part_iii[5, 6] = + 2.
        part_iii[7, 8] = + 4.
        part_iii[8, 7] = - 4.
        part_iii /= self.gamma

        part_iii = _kron(
            np.diag(
                self.momentums
            ),
            part_iii
        )

        #####

        part_iv = np.zeros((9, 9))
        part_iv[0, 4] = part_iv[0, 6] = -self.s/2**.5
        part_iv[1, 4] = +self.s / 2 ** .5
        part_iv[1, 1] = -1
        part_iv[2, 6] = +self.s / 2 ** .5
        part_iv[2, 2] = -1

        part_iv[3, 3] = part_iv[4, 4] = part_iv[5, 5] = part_iv[6, 6] = -.5
        a = self.delta - 1/self.gamma
        part_iv[3, 4] = part_iv[5, 6] = +a
        part_iv[4, 3] = part_iv[6, 5] = -a

        part_iv[3, 8] = -self.s / 2 ** 1.5
        part_iv[4, 7] = part_iv[4, 1] = -self.s / 2 ** 1.5
        part_iv[4, 0] = self.s / 2 ** 1.5

        part_iv[5, 8] = self.s / 2 ** 1.5
        part_iv[6, 7] = part_iv[6, 2] = -self.s / 2 ** 1.5
        part_iv[6, 0] = self.s / 2 ** 1.5

        part_iv[7, 7] = part_iv[8, 8] = -1.
        part_iv[7, 4] = part_iv[7, 6] = part_iv[8, 3] = self.s / 2 ** 1.5
        part_iv[8, 5] = -self.s / 2 ** 1.5

        part_iv = _kron(np.eye(self.momentum_states), part_iv)

        return part_i + part_ii + part_iii + part_iv

    def evolve(self, tot_steps, time_step):
        """Integrate the equation of motion 
        """

        self.state = np.zeros(self.momentum_states * 9)
        self.state[1::9] = np.exp(-(np.arange(-self.momentum_per_recoil*self.max_momentum,
                                              self.momentum_per_recoil*self.max_momentum) / self.momentum_per_recoil/.2)**2)
        self.state[1::9] /= np.sum(self.state[1::9])

        step_mat = time_step * self.m
        for _ in tqdm(range(tot_steps)):
            self.state += step_mat @ self.state

        plt.plot(self.momentums, self.state[::9])
        plt.plot(self.momentums, self.state[1::9])
        plt.plot(self.momentums, self.state[2::9])
        plt.plot(self.momentums, self.get_momentum_distribution(
            self.state.reshape(-1, 9)))
        plt.show()

    def get_momentum_distribution(self, full_distribution):
        r"""Solve for momentum distribution 

        Parameter:
        ----------
        - full_distribution : a (momentum_states \times 9) matrix; 
        """

        ret = np.array(full_distribution[:, 0])
        ret[self.momentum_per_recoil:] += full_distribution[:-self.momentum_per_recoil, 1]
        ret[:-self.momentum_per_recoil] += full_distribution[self.momentum_per_recoil:, 2]
        return ret

    def solve_full_distribution(self):
        first_row = np.array(self.m[0])
        self.m[0] = 0.
        self.m[0, ::9] = 1
        self.m[0, 1::9] = 1
        self.m[0, 2::9] = 1

        y = np.zeros(self.momentum_states * 9)
        y[0] = 1
        ret = np.linalg.solve(self.m, y).reshape(-1, 9)

        self.m[0] = first_row

        return ret

    def eig_show(self):
        """Plot the eigen-functions corresponding the 10 largest eigenvalues 
        """

        w, v = np.linalg.eig(self.m)
        for i in reversed(np.argsort(w.real)[-10:]):
            plt.plot(
                self.momentums, v[:, i].real.reshape(-1, 9)[:, 0], label='%.3e' % w[i].real)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(linewidth=120, precision=2)
    # sol = Solver(2, -2.5, nmax=(5, 10), gamma=10).eig_show()
    sol = Solver(2, -2.5, nmax=(10, 20), gamma=1.6, r=float("inf"))
    # sol.evolve(100000, .0001)
    plt.plot(sol.momentums, sol.get_momentum_distribution(
        sol.solve_full_distribution()))
    plt.show()
    sol = Solver(1, -2.5, nmax=(10, 30), gamma=1, r=float("inf"))
    # sol.evolve(100000, .0001)
    plt.plot(sol.momentums, sol.get_momentum_distribution(
        sol.solve_full_distribution()))

    plt.axhline(0)
    plt.show()
