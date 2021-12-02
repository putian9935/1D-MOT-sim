__doc__ = """Solve for stationary momentum distribution of J=0 <-> J=1 transition. 

The full master equation will be solved (in a subspace of diagonal momentums) and the results should match. 
"""

import numpy as np
from functools import lru_cache


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
        idx = 0
        part_ia = np.zeros((self.momentum_states, self.momentum_states))
        part_ib = np.zeros((self.momentum_states, self.momentum_states))

        tabbed = np.hstack([self.n_u,[0] * (self.momentum_states - len(self.n_u))])
        for i in range(-self.max_momentum, self.max_momentum):
            for j in range(self.momentum_per_recoil):
                part_ia[idx,:] = np.roll(tabbed,-len(self.n_u)+idx+1)
                # 1.a \pi_+(u-1)
                # left = max(0, idx - self.momentum_per_recoil * 2)
                # right = idx + 1
                # if right > (left ):
                #     part_ia[idx, left:right] = self.n_u[-(right-left):]

                part_ib[idx,:] = np.roll(tabbed,idx-1)
                # 1.b \pi_-(u+1)
                # left = idx
                # right = min(self.momentum_states, idx +
                #             self.momentum_per_recoil * 2)
                # if right > (left ):
                #     part_ib[idx, left:right] = self.n_u[:(right-left)]

                idx += 1



        buf = np.zeros((9, 9))

        buf[0, 1] = 1.
        part_i = np.kron(part_ia, buf)
        buf[0, 1] = 0.

        buf[0, 2] = 1.
        part_i += np.kron(part_ib, buf)

        part_i /= self.momentum_per_recoil

        #####

        part_ii = np.diag([1]*(self.momentum_states-1), 1) - \
            np.diag([1]*(self.momentum_states-1), -1)
        part_ii = np.kron(part_ii, np.eye(9)) / (2. * self.r)

        #####

        part_iii = np.zeros((9, 9))
        part_iii[3, 4] = - 2.
        part_iii[4, 3] = + 2.
        part_iii[5, 6] = - 2.
        part_iii[6, 5] = + 2.
        part_iii[7, 8] = + 4.
        part_iii[8, 7] = - 4.
        part_iii /= self.gamma

        part_iii = np.kron(
            np.diag(
                np.arange(-self.momentum_per_recoil*self.max_momentum,
                          self.momentum_per_recoil*self.max_momentum) / self.momentum_per_recoil
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
        part_iv[3, 4] = part_iv[5, 6] = a
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

        part_iv = np.kron(np.eye(self.momentum_states), part_iv)

        return part_i + part_ii + part_iii + part_iv

    def solve_distribution(self):
        
        new_row = np.zeros(9 * self.momentum_states)
        new_row[::9] = 1
        new_row[1::9] = 1
        new_row[2::9] = 1

        y = np.zeros(self.momentum_states * 9)
        y[0] = 1
        self.m[0]  = new_row
        ans = np.linalg.solve(self.m, y).reshape(self.momentum_states,9)
        
        # append_mat = np.vstack([self.m, new_row])
        # ans = (np.linalg.pinv(append_mat) @ y).reshape(self.momentum_states, 9)
        plt.plot(ans[:, 0])
        plt.plot(ans[:, 1])
        plt.plot(ans[:, 2])
        plt.show()

    def eig_play(self):
        w, v= np.linalg.eig(self.m)
        largest = v[:, np.argmax(w.real)].real.reshape(self.momentum_states, 9)[:,0]
        # plt.plot(w.real, w.imag, "+")
        # plt.show()
        
        for i in reversed(np.argsort(w.real)[-10:]):
            plt.plot(v[:,i].real.reshape(self.momentum_states, 9)[:,0])
        plt.show()
        
np.set_printoptions(linewidth=120)
Solver(2, 0, nmax=(20, 5), r=float("inf")).solve_distribution()
