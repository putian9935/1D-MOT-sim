__doc__ = """The MCWF program for simulating 1D MOT. 
"""


import numpy as np
from numpy.lib.arraysetops import isin
from cg_coefficient import cg, cg_modulus2
from tqdm import tqdm
import numba as nb
from collections.abc import Iterable


@nb.vectorize(fastmath=True)
def abs2(x):
    return x.real ** 2 + x.imag ** 2


@nb.jit(fastmath=True)
def vec_mod2(x):
    return sum(abs2(x))


@nb.jit
def normalize(x):
    x /= np.vdot(x, x).real ** .5


class Simulator:
    """Simulator of 1D MOT with MCWF 

    Conventions
    -----------
    1. (-, 0, +) -> (0, 1, 2)
    2. In epsilon(polarization), the first component is propagating to right, and the second to left 
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

        self.p_bar_t_flattened = self.gamma * np.array([
            [1/5, 1/10, 1/5],
            [3/5, 4/5, 3/5],
            [1/5, 1/10, 1/5]
        ]).T.flatten()
        self.probs = np.zeros(9)

        self.hamiltonian = self.calc_hamiltonian()

        self.c_matrices = self.c_matrix()

        self.p_mat = np.kron(
            np.diag(list(range(-self.max_momentum, self.max_momentum+1))),
            np.eye(self.spin_states)
        )
        self.p2_mat = np.kron(
            np.diag(list(range(-self.max_momentum, self.max_momentum+1)))**2.,
            np.eye(self.spin_states)
        )

        self.ranges = tuple(
            list(
                range(-min(self.je, self.jg-q), min(self.je, self.jg+q)+1)
            ) for q in (-1, 0, 1)
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
                        if not ipz:  # propagating to the right
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
        The *k* indices goes first, meaning the first three entries correspond to q=-1
        """

        self.probs[:3] = self.p_bar_t_flattened[:3] * \
            self.calc_probability_angular(-1)
        self.probs[3:6] = self.p_bar_t_flattened[3:6] * \
            self.calc_probability_angular(0)
        self.probs[6:] = self.p_bar_t_flattened[6:] * \
            self.calc_probability_angular(1)

    def calc_probability_angular(self, q):
        """Returns the sum part of probability"""
        ret = 0
        for me in self.ranges[q+1]:
            ret += cg_modulus2(1, q, self.jg, me-q, self.je, me) * vec_mod2(
                self.state[self.ground_states_offset + me + self.je::self.spin_states])

        return ret

    def jump(self, time_step):
        """Determine whether a jump is made and, if yes, and where it is to. 

        Returns
        -------
        state: a number of 0 to 9 
            for state 0 - 8, the corresponding jump is made; 
            for state 9, no jump is made 
        """

        cur = 0.
        u = np.random.random() / time_step
        self.calc_probability()
        for i, p in enumerate(self.probs):
            cur += p
            if cur > u:
                return i
        return 9

    def simulate(self, tot_steps, time_step, stat_funcs=None, every_n_save=100, init_state=None, init_params=None):
        """The MCWF simulator. 

        Parameters
        ----------
        stat_funcs : a list of functions for statistics
        init_state : function to initialize state, takes jg, je, and nmax as its first 3 parameters 
        init_params : other parameters to pass to init_state

        Returns 
        -------
        statistics after each time step 
        """

        if isinstance(init_state, type(None)):
            init_state = Simulator.gaussian_init
        if isinstance(init_params, type(None)):
            init_params = ()
        elif not isinstance(init_params, Iterable):
            init_params = (init_params,)

        self.state = init_state(
            self.jg, self.je, self.max_momentum, *init_params)

        ts_ham = -1j * time_step * self.hamiltonian
        ret = []
        for _ in tqdm(range(tot_steps)):
            jump = self.jump(time_step)
            if jump == 9:  # no jump
                self.state += ts_ham @ self.state
            else:
                self.state = self.c_matrices[jump] @ self.state

            # use np.sum if larger dimension
            normalize(self.state)

            if not _ % every_n_save:
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
            es_dagger[:(2*self.jg+1), (2*self.jg+1)                      :] = np.conjugate(np.transpose(self.es_matrix(q)))
            for ik, k in enumerate((-1, 0, 1)):
                ret.append(
                    self.p_bar[ik, iq] ** .5 * np.kron(
                        np.diag([1]*(2*self.max_momentum + 1 - abs(k)), -k),
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

    def momentum2(self):
        """Returns average squared momentum 

        Returns
        -------
        The average squared momentum 
        """
        return (np.conjugate(self.state).T @ self.p2_mat @ self.state)

    def ground_state(self):
        """Returns ground state probability

        Returns
        -------
        The ground state probability
        """
        return np.sum((np.conjugate(self.state).T * self.state)[::self.spin_states])

    @staticmethod
    def gaussian_init(je, jg, nmax):
        """Generate a gaussian wavepacket
        """
        ret = np.zeros(2*(je+jg+1)*(2*nmax+1), dtype=np.complex128)
        sigma = nmax / 4.
        ret[1::2*(je+jg+1)] = np.exp(-np.arange(-nmax, nmax+1)**2/2/sigma**2)
        normalize(ret)
        return ret

    @staticmethod
    def plane_wave_init(je, jg, nmax, init_p=None):
        """Generate a plane wave state
        """
        if isinstance(init_p, type(None)):
            init_p = nmax // 2
        ret = np.zeros(2*(je+jg+1)*(2*nmax+1), dtype=np.complex128)
        ret[-init_p * 2*(je+jg+1)] = 1.
        return ret 
