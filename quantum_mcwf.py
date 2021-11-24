__doc__ = """The MCWF program for simulating 1D MOT. 
"""


import numpy as np
from cg_coefficient import cg


class Simulator:
    """Simulator of 1D MOT with MCWF 

    Convention: (-, 0, +) -> (0, 1, 2)
    """

    def __init__(self, s,  delta, b, gamma=1.6, epsilon=(1, 0, 0), jg=0, je=1, nmax=100):
        self.jg = jg
        self.je = je
        self.spin_states = 2 * (jg + je + 1)
        self.ground_states_offset = 2 * jg + 1  # offset in terms of index
        self.n = self.spin_states * (2*nmax+1)
        self.state = np.zeros(self.n, dtype=np.complex128)

        # first index: k; second index: q
        self.bar_p = np.array(
            [[1/5, 1/10, 1/5], [3/5, 4/5, 3/5], [1/5, 1/10, 1/5]]) ** .5

        self.s = s
        self.delta = delta
        self.gamma = gamma
        self.b = b

        self.max_momentum = nmax
        self.polarization = epsilon

        # matrix p, row is k; column is q
        self.p_bar = self.gamma * np.array([
            [1/5, 1/10, 1/5],
            [3/5, 4/5, 3/5],
            [1/5, 1/10, 1/5]
        ])

        self.hamiltonian = self.calc_hamiltonian()
        self.state = np.ones(self.n, dtype=np.complex128)
        print(self.calc_probability().flatten())

    def calc_hamiltonian(self):
        """Returns hamiltonian
        """
        ret = np.zeros((self.n, self.n), dtype=np.complex128)

        momentum = np.kron(np.diag(
            np.arange(-self.max_momentum, self.max_momentum+1)**2), np.eye(self.spin_states))

        light = np.zeros((self.n, self.n), dtype=np.complex128)
        for iq, q in enumerate((-1, 0, 1)):
            if not self.polarization[iq]:
                continue
            for ie, me in enumerate(range(-self.je, self.je+1)):
                mg = me - q
                if not -self.jg <= mg <= self.jg:
                    continue  # constraint of CG coeff
                ig = mg + self.jg
                for n, _ in enumerate(range(-self.max_momentum, self.max_momentum+1)):
                    if not n:
                        continue
                    light[n*self.spin_states+ie+self.ground_states_offset, (n-1)*self.spin_states+ig] = cg(
                        self.jg, mg, 1, q, self.je, me) * self.polarization[iq]
        light += np.conjugate(np.transpose(light))

        excited_state_projector = np.kron(
            np.eye(2*self.max_momentum+1), np.diag([0]*(2*self.jg+1)+[1]*(2*self.je+1)))

        mag_field = np.kron(np.diag([1]*(2*self.max_momentum), -1)-np.diag(
            [1]*(2*self.max_momentum), 1), np.eye(self.spin_states))
        ret = (momentum +  # free-evolution
               self.gamma * self.s**.5 / 2. * light -  # atom-field interaction
               # detuning & Lindblad
               self.gamma*(.5j+self.delta) * excited_state_projector +
               .5j*self.b * mag_field)  # magnetic field gradient

        return ret

    def calc_probability(self):
        r"""Returns the *flattened* p_{k,q} / \Delta \tau

        Note
        ----
        The q indices goes first, meaning the first three entries correspond to k=-1
        """
        ret = np.array(self.p_bar)
        for iq, q in enumerate((-1, 0, 1)):
            ret[:, iq] *= self.calc_probability_angular(q)
        return ret.flatten()

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
        
        distribution = np.hstack((0,time_step * self.calc_probability()))
        distribution.cumsum(out=distribution)
        return np.searchsorted(distribution, np.random.random()) - 1

    def simulate(self, tot_steps, time_step, stat_funcs=None):
        """The MCWF simulator. 
        """
        for _ in range(tot_steps):
            if self.jump(time_step) == 9:
                self.state += time_step * self.hamiltonian @ self.state  
            else:
                pass
    

    def es_matrix(self, q):
        r"""The matrix epsilon_q \cdot S+

        Returns
        -------
        A matrix of size (2*je+1)\times(2*jg+1)
        """
        ret = np.zeros((2*self.je+1, 2*self.jg+1))
        for i, me in enumerate(range(-self.je, self.je+1)):
            mg = me - q
            if -self.jg <= mg <= self.jg:
                ret[i, mg+self.jg] = cg(1, q, self.jg, mg, self.je, self.me)
        return ret


np.set_printoptions(linewidth=180)
Simulator(1, 1, 2, nmax=1).simulate(100,.01)
