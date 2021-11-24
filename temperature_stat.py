import numpy as np 
import matplotlib.pyplot as plt 

## detuning 
var = 'maggrad'
data = np.genfromtxt('%s1D.csv'%var, delimiter = ',')

plt.plot(data[7:,0], data[7:,1]*2.82624272560609e-01,'+')
plt.xlabel('$b$')
plt.ylabel('$T\,/\,\mu\mathrm{K}$')
plt.show()

