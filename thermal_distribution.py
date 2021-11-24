import numpy as np
import matplotlib.pyplot as plt


from scipy.constants import value

@np.vectorize
def f(u):
    return np.exp(-(u/ut)**2/2)


ut = 59

uc = 39

us = np.linspace(0, 200)

plt.plot(us, f(us))
plt.axvline(uc, c='r')
plt.ylim(bottom=0)
plt.xlim(0, 200)
plt.xlabel('$u$')
plt.ylabel('$p(u)\,/\,\mathrm{a.u.}$')
plt.show()