import matplotlib.pyplot as plt
__doc__ = """The MCWF program for simulating 1D MOT. 
"""


import numpy as np
from cg_coefficient import cg
from tqdm import tqdm


class Simulator:
    """Simulator of 1D MOT with MCWF 

    Convention: (-, 0, +) -> (0, 1, 2)
    """

    def __init__(self, s,  delta, b, gamma=1.6, epsilon=None, jg=0, je=1, gg=0, ge=1.5, nmax=100):
        self.jg = jg
        self.je = je
        self.gg = gg
        self.ge = ge
        self.spin_states = 2 * (jg + je + 1)
        self.ground_states_offset = 2 * jg + 1  # offset in terms of index
        self.n = self.spin_states * (2*nmax+1)
        self.state = np.zeros(self.n, dtype=np.complex128)
        self.state[0] = 1

        # first index: k; second index: q
        self.bar_p = np.array(
            [[1/5, 1/10, 1/5], [3/5, 4/5, 3/5], [1/5, 1/10, 1/5]]) ** .5

        self.s = s
        self.delta = delta
        self.gamma = gamma
        self.b = b

        self.max_momentum = nmax
        if isinstance(epsilon, type(None)):
            self.polarizations = [(1, 0, 0), (0, 0, 1)]
        else:
            self.polarizations = epsilon

        # matrix p, row is k; column is q
        self.p_bar = self.gamma * np.array([
            [1/5, 1/10, 1/5],
            [3/5, 4/5, 3/5],
            [1/5, 1/10, 1/5]
        ])

        self.hamiltonian = self.calc_hamiltonian()

        self.c_matrices = self.c_matrix()

        self.p_mat = np.kron(
            np.diag(list(range(-self.max_momentum, self.max_momentum+1))),
            np.eye(self.spin_states)
        )

    def calc_hamiltonian(self):
        """Returns hamiltonian
        """
        ret = np.zeros((self.n, self.n), dtype=np.complex128)

        momentum = np.kron(np.diag(
            np.arange(-self.max_momentum, self.max_momentum+1)**2), np.eye(self.spin_states))

        light = np.zeros((self.n, self.n), dtype=np.complex128)
        for ipz, pz in enumerate(self.polarizations):
            for iq, q in enumerate((-1, 0, 1)):
                if not pz[iq]:
                    continue
                for ie, me in enumerate(range(-self.je, self.je+1)):
                    mg = me - q
                    if not -self.jg <= mg <= self.jg:
                        continue  # constraint of CG coeff
                    ig = mg + self.jg
                    for n, _ in enumerate(range(-self.max_momentum, self.max_momentum+1)):
                        if not n:
                            continue
                        if not ipz:
                            light[n*self.spin_states+ie+self.ground_states_offset,
                                  (n-1)*self.spin_states+ig] = pz[iq] * cg(self.jg, mg, 1, q, self.je, me)
                        else:
                            light[(n-1)*self.spin_states+ie+self.ground_states_offset,
                                  n*self.spin_states+ig] = pz[iq] * cg(self.jg, mg, 1, q, self.je, me)
        light += np.conjugate(np.transpose(light))

        excited_state_projector = np.kron(
            np.eye(2*self.max_momentum+1),
            np.diag([0]*(2*self.jg+1)+[1]*(2*self.je+1))
        )

        mag_field = np.kron(
            np.diag([1]*(2*self.max_momentum), -1) -
            np.diag([1]*(2*self.max_momentum), 1),
            np.diag([self.gg * mg for mg in range(-self.jg, self.jg+1)] +
                    [self.ge * me for me in range(-self.je, self.je+1)])
        )
        ret = (momentum +  # free-evolution
               # atom-field interaction
               self.gamma * self.s**.5 / 2. * light +
               # detuning & Lindblad
               -self.gamma*(.5j+self.delta) * excited_state_projector +
               .5j*self.b * mag_field)  # magnetic field gradient

        return ret

    def calc_probability(self):
        r"""Returns the *flattened* p_{k,q} / \Delta \tau

        Note
        ----
        The *k* indices goes first, meaning the first three entries correspond to k=-1
        """
        ret = np.array(self.p_bar)
        for iq, q in enumerate((-1, 0, 1)):
            ret[:, iq] *= self.calc_probability_angular(q)

        return ret.T.flatten()

    def calc_probability_angular(self, q):
        """Returns the sum part of probability"""
        ret = 0
        for ie, me in enumerate(range(-self.je, self.je+1)):
            mg = me - q
            if not -self.jg <= mg <= self.jg:
                continue
            cg_mod2 = abs(cg(1, q, self.jg, mg, self.je, me))**2
            state_mod2 = 0
            for ip, _ in enumerate(range(-self.max_momentum, self.max_momentum + 1)):
                state_mod2 += abs(self.state[ip*self.spin_states +
                                  self.ground_states_offset + ie])**2
            ret += cg_mod2 * state_mod2
        return ret

    def jump(self, time_step):
        """Determine whether a jump is made and, if yes, and where it is to. 

        Returns
        -------
        state: a number of 0 to 9 
            for state 0 - 8, the corresponding jump is made; 
            for state 9, no jump is made 
        """

        distribution = np.hstack((0, time_step * self.calc_probability()))
        distribution.cumsum(out=distribution)
        return np.searchsorted(distribution, np.random.random()) - 1

    def simulate(self, tot_steps, time_step, stat_funcs=None):
        """The MCWF simulator. 

        Returns 
        -------
        statistics after each time step 
        """
        ret = []
        for _ in tqdm(range(tot_steps)):
            jump = self.jump(time_step)
            if jump == 9:  # no jump
                self.state += -1j*time_step * self.hamiltonian @ self.state
            else:
                self.state = self.c_matrices[jump] @ self.state

            self.state /= np.sum(np.abs(self.state)**2)**.5

            new_entry = [jump]
            for func in stat_funcs:
                new_entry.append(func(self))

            ret.append(new_entry)

        return np.array(ret, dtype=np.complex128)

    def c_matrix(self):
        """The c matrix divided by time step. The constant gamma is omitted

        Returns
        -------
        a list of c matrices

        Note
        ----
        This function is called just once 
        """
        ret = []
        for iq, q in enumerate((-1, 0, 1)):
            es_dagger = np.zeros(
                (self.spin_states, self.spin_states), dtype=np.complex128)
            es_dagger[:(2*self.jg+1), (2*self.jg+1):] = np.conjugate(np.transpose(self.es_matrix(q)))
            for ik, k in enumerate((-1, 0, 1)):
                ret.append(
                    self.p_bar[ik, iq] ** .5 * np.kron(
                        np.diag([1]*(2*self.max_momentum + 1 - abs(k)), k),
                        es_dagger
                    )
                )

        return ret

    def es_matrix(self, q):
        r"""The matrix epsilon_q \cdot S+

        Returns
        -------
        A matrix of size (2*je+1)\times(2*jg+1)
        """
        ret = np.zeros((2*self.je+1, 2*self.jg+1), dtype=np.complex128)
        for i, me in enumerate(range(-self.je, self.je+1)):
            mg = me - q
            if -self.jg <= mg <= self.jg:
                ret[i, mg+self.jg] = cg(1, q, self.jg, mg, self.je, me)
        return ret

    def momentum(self):
        """Returns average momentum 

        Returns
        -------
        The average momentum 
        """
        return (np.conjugate(self.state).T @ self.p_mat @ self.state)

    def ground_state(self):
        """Returns ground state probability

        Returns
        -------
        The ground state probability
        """
        return np.sum((np.conjugate(self.state).T * self.state)[::self.spin_states])


np.set_printoptions(linewidth=180)
result = Simulator(
    1, -10, 0, nmax=10, epsilon=[(0, 0, 1), (1, 0, 0)]).simulate(10000, .1, [Simulator.momentum, Simulator.ground_state])


print(*np.unique(result[:, 0].real, return_counts=True))
plt.plot(*np.unique(result[:, 0].real, return_counts=True), "+")
plt.show()
plt.plot(result[:, 1].real)
plt.show()
plt.plot(result[:, 2].real)
plt.show()
