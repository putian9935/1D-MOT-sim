__doc__ = """
Calculates the constants for a given simulation. 
"""

from scipy.constants import *
import math

from scipy.constants.constants import Boltzmann   # for fabs

############# Start of input ############
# using strontium-88
mass = 88 * value('atomic mass constant')

# wavelength
wl = 689e-9

# nature linewidth
gamma = 2 * pi * 7.5e3

# magnetic field gradient
B_grad = 10 * 1e-2

# g-factor between levels
g = 1.5

# detuning in Hz
detuning = -120e3

# normalized intensity
s = 284

# temperature
temp = 1e-3

############# End of input ############

k = 2 * pi / wl
omega_r = hbar * k * k / (2. * mass)

v0 = gamma / k
x0 = gamma / (k * omega_r)
b = g * value('Bohr magneton') / hbar * B_grad / (omega_r * k)

print('Velocity unit v_0 = %.3e' % v0)
print('Displacement unit x_0 = %.3e' % x0)
print('Position coefficient b = %.3e' % b)

delta = 2 * pi * detuning / gamma
print('Detuning delta = %.3e' % delta)

zeta = (4*math.fabs(delta)*s/b/(1+s+4*delta*delta)**2) ** .5
print('Damping ratio zeta = %.3e' % zeta)


print('Normalized thermal velocity is', (Boltzmann * temp / mass)**.5 / v0)

print('Temperature is', mass * v0 ** 2 / Boltzmann)
