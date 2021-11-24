__doc__ = """
Investigate the capture velocity of 1D MOT. 
"""



from scipy.integrate import solve_ivp
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
        def force(x, u):
            tmp = u + b * x
            return 16 * s * delta * tmp / (1+s+4*(delta - tmp)**2) / (1+s+4*(delta + tmp)**2)
        self.force = force

    def simulate_position(self, x0, u0, d, t_span=None):
        """
        Parameters: 
        x0: initial position (assuming atom coming from left, this is taken as the left of the beam)
        u0: initial velocity 
        d: beam width (beam is assumed to center at 0)
        t_span: integral range 

        Returns: 
        A 1d numpy.ndarray containing position at given time defined in t_span
        """

        if t_span is None:
            t_span = np.linspace(0, 3000, 1000)

        def truncated_force(x, u):
            return self.force(x, u) if -d < x < d else 0.

        # solve initial value problem with RK45 (MATLAB ode45 if the name sounds more familiar)
        sol = solve_ivp(
            # the first component is time
            lambda _, y: (y[1], truncated_force(*y)),
            (min(t_span), max(t_span)),  # the lower/upper limit of integration
            (x0, u0),  # the initial condition
            dense_output=True,  # need position at every given time
            first_step=1, max_step=10  # some fine tune which may not be necessary
        )

        return sol.sol(t_span)[0]

    def simulate_convergence(self, x0, u0, d, t_span=None):
        """
        Parameters: 
        x0: initial position (assuming atom coming from left, this is taken as the left of the beam)
        u0: initial velocity 
        d: beam width (beam is assumed to center at 0)
        t_span: integral range 

        Returns: 
        Whether the atom is captured or not; 
        Determined by whether the final position of the atom lies inside the beam or not 
        """

        if t_span is None:
            t_span = (0, 3000)

        def truncated_force(x, u):
            return self.force(x, u) if -d < x < d else 0.

        sol = solve_ivp(
            lambda _, y: (y[1], truncated_force(*y)),
            t_span,
            (x0, u0),
            first_step=1, max_step=30
        )
        return np.abs(sol.y[0, -1]) < d

    def simulate_full(self, x0, u0, d, t_span=None):
        """
        Parameters: 
        x0: initial position (assuming atom coming from left, this is taken as the left of the beam)
        u0: initial velocity 
        d: beam width (beam is assumed to center at 0)
        t_span: integral range 

        Returns: 
        A 1d numpy.ndarray containing position & velocity at final time
        """

        if t_span is None:
            t_span = np.linspace(0, 3000, 3000)

        def truncated_force(x, u):
            return self.force(x, u) if -d < x < d else 0.

        # solve initial value problem with RK45 (MATLAB ode45 if the name sounds more familiar)
        sol = solve_ivp(
            # the first component is time
            lambda _, y: (y[1], truncated_force(*y)),
            (min(t_span), max(t_span)),  # the lower/upper limit of integration
            (x0, u0),  # the initial condition
            dense_output=True,
            first_step=1, max_step=5  # some fine tune which may not be necessary
        )

        return sol.sol(t_span)

    def simulate_final(self, x0, u0, d, t_span=None):
        """
        Parameters: 
        x0: initial position (assuming atom coming from left, this is taken as the left of the beam)
        u0: initial velocity 
        d: beam width (beam is assumed to center at 0)
        t_span: integral range 

        Returns: 
        A 1d numpy.ndarray containing position & velocity at final time
        """

        if t_span is None:
            t_span = (0, 3000)

        def truncated_force(x, u):
            return self.force(x, u) if -d < x < d else 0.

        # solve initial value problem with RK45 (MATLAB ode45 if the name sounds more familiar)
        sol = solve_ivp(
            # the first component is time
            lambda _, y: (y[1], truncated_force(*y)),
            (min(t_span), max(t_span)),  # the lower/upper limit of integration
            (x0, u0),  # the initial condition
            first_step=1, max_step=5  # some fine tune which may not be necessary
        )

        return sol.y[:, -1]

    def calc_capture_velocity(self, d, t, ul=0, ur=80):
        """Calculate capture velocity given diameter and time range with binary search. 
        
        Parameters
        ----------
        d : beam diameter 
        t : time range 
        ul : lower guess of capture velocity 
        ur : upper guess of capture velocity 

        Returns
        -------
        The estimated capture velocity
        """

        while (ur - ul) / (ur + ul) > 5e-4:
            u0 = (ur + ul) / 2.
            if self.simulate_convergence(-d, u0, d, t):
                ul = u0
            else:
                ur = u0
        return (ur + ul) / 2.

    def plot_capture_velocity(self, diams, loglog=False):
        """Plot the capture velocity versus beam diameter. 

        Parameters
        ----------
        diams : an 1d array of beam diameters from which capture velocity is calculated
        loglog : a flag of whether to show result in log-log scale 

        Recommended parameters
        ----------------------
        Blue MOT: 

        Red MOT: 
        diams = np.arange(1,40,.5)
        """

        t = (0, 500)
        u0s = []
        for d in tqdm(diams):
            u0s.append(self.calc_capture_velocity(d, t))
        plt.figure()
        bottom = plt.axes([.12, .12, .76, .2])

        bottom.plot(diams, self.force(diams, 0))
        bottom.set_xlabel('$y$')
        bottom.set_ylabel('$F(y,u=0)$')
        top = plt.axes([.12, .4, .76, .5])
        plt.plot(diams, u0s)
        plt.ylabel('$u_c$')

        if loglog:
            top.set_xscale('log')
            top.set_yscale('log')
        plt.show()

    def plot_trace(self, u0, d):
        """Plot the real space trace. 

        Parameters
        ----------
        u0 : initial velocity
        d : diameter of the beam  
        """

        plt.figure(figsize=(6, 12))

        # bottom panel for force
        bottom = plt.axes([.12, .12, .76, .2])
        x = np.linspace(-1500, 1500, 1000)
        bottom.plot(x, model.force(x, 0))
        bottom.set_xlabel('$y$')
        bottom.set_ylabel('$F(y,u=0)$')
        bottom.add_patch(Rectangle((-d, -3), 2*d, 6,
                         facecolor='orange', alpha=.6))
        bottom.set_ylim(-1, 1)

        # top panel for trace
        plt.axes([.12, .4, .76, .5])
        t = np.linspace(0, 500, 1000)
        x0 = model.simulate_position(-d, u0, d)
        plt.plot(x0, t, label='$x_0=%d$' % (-d))
        plt.ylabel(r'$\tau$')
        plt.legend()

        plt.title('$u_0=%.1f$' % u0)
        plt.axvline(0, c='k')
        plt.xlim(np.min(x), np.max(x))
        plt.show()

    def plot_velocity_versus_position(self, d, u0s, t_span):
        """A Zeeman-slower-like performance plot 

        Parameters
        ----------

        d : beam diameter 
        u0s : an array of initial velocities 
        t_span : an array of time points. Finer this array leads to finer details of the plot 
        """

        # related alpha for aesthetic
        alphas = (np.max(u0s)-u0s)/np.ptp(u0s) * .8 + .2
        for u0, alpha in tqdm(zip(u0s, alphas)):
            res = model.simulate_full(-d * 1.1, u0, d, t_span)
            plt.plot(res[0], res[1], alpha=alpha, c='r')
        plt.xlim(-d * 1.2, d * 1.1)
        plt.axvline(-d, linestyle='--', c='b')
        plt.axvline(d, linestyle='--', c='b')
        plt.xlabel('$y$')
        plt.ylabel('$u$')
        plt.title('$d=%d$' % d)
        plt.show()

    def plot_velocity_distribution(self, x0, left, right, ut, d, single_side=True, int_time=1000, sample_size=100000):
        """Plot the final velocity against initial velocity, and the distributions. 

        Parameters
        ----------
        x0 : initial position assumed at t=0
        left : the lower limit of velocity range, only those inside are considered 
        right : the upper limit of velocity range, only those inside are considered 
        ut : the thermal velocity at input; the initial velocity is assumed to follow (absolute) Gaussian
        d : the beam diameter 
        single_side : whether or not the initial distribution is absolute or not 
        int_time : the integral time 
        """
        u0s = np.linspace(left, right, 400)
        uf = []
        for u0 in tqdm(u0s):
            uf.append(self.simulate_final(x0, u0, d, (0, int_time))[1])

        # the velocity dependece
        plt.figure()
        plt.plot(u0s, uf, label='With MOT')
        plt.plot(u0s, u0s, '--', label='Without MOT')

        plt.xlabel('$u_0$')
        plt.ylabel('$u_f$')
        plt.legend()
        plt.show()

        interp = Akima1DInterpolator(u0s, uf)
        ufi = np.vectorize(lambda _: interp(_) if left < _ < right else np.nan)

        u0_sample = np.random.normal(0, ut, sample_size)
        if single_side:
            u0_sample = np.abs(u0_sample)

        uf_sample = ufi(u0_sample)

        # the velocity distribution
        plt.figure()
        plt.hist(u0_sample, density=True, bins=100)  # might change this to 50
        plt.hist(uf_sample, density=True, bins=100, alpha=.8)
        plt.xlim(left, right)
        plt.xlabel('$u$')
        plt.ylabel('$p(u)\,/\,\mathrm{a.u.}$')
        plt.show()


def plot_detuning_dependence(deltas):
    vels = [MOT1DModel(284, delta, .048).calc_capture_velocity(
        900, (0, 1000)) for delta in tqdm(deltas)]

    plt.plot(deltas, vels)
    plt.xlabel('$\delta$')
    plt.ylabel('$u_c$')
    plt.show()


def plot_intensity_dependence(ss):
    vels = [MOT1DModel(s, -16, .048).calc_capture_velocity(900, (0, 1500))
            for s in tqdm(ss)]

    plt.plot(ss, vels)
    plt.xlabel('$s$')
    plt.ylabel('$u_c$')
    plt.show()


def plot_mag_grad_dependence(bs):
    vels = [MOT1DModel(284, -16, b).calc_capture_velocity(900, (0, 1500))
            for b in tqdm(bs)]
    plt.plot(bs, vels, label='$\delta=-16$')

    vels = [MOT1DModel(284, -30, b).calc_capture_velocity(900, (0, 1500))
            for b in tqdm(bs)]
    plt.plot(bs, vels, label='$\delta=-30$')
    plt.xlabel('$b$')
    plt.ylabel('$u_c$')
    plt.legend()
    plt.show()


# Below I list some common usage and calls 

model = MOT1DModel(284, -16, .048)  # this is for the red MOT

# Plot trace
# model.plot_trace(35,900)

# Plot capture velocity
# model.plot_capture_velocity(np.exp(np.linspace(2, 7.5, 200)))

# Detuning dependence
# plot_detuning_dependence(-np.arange(1, 100))

# Intensity dependence
# plot_intensity_dependence(np.linspace(0, 2000, 100))

# Magnetic field gradient dependence
# plot_mag_grad_dependence(np.linspace(0, .15, 100))

# Zeeman-slower-like plot
# model.plot_velocity_versus_position(900, np.arange(1, 50), np.linspace(0, 3000, 3000))

# Plot velocity distribution
# model.plot_velocity_distribution(-900, 0, 200, 59, 900)

# Reproduce Fig. 15 of Loftus' paper
# MOT1DModel(30, 20, 0.).plot_velocity_distribution(0, -40, 40, 6.38, 900, single_side=False, int_time=750, sample_size=500000)
