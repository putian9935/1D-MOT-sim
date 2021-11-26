__doc__ = """Calculate the matrix Xi from eq. (4.24)

Remark
------
For the case of red MOT(Je=1, Jg=0), using CG coefficient is evidently an overkill. 
"""

from sympy.physics.quantum.cg import CG 
import numpy as np 

from functools import lru_cache

@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, j, m):
    """CG coefficient"""
    return complex(CG(j1, m1, j2, m2, j, m).doit())

@lru_cache(maxsize=None)
def cg_modulus2(j1, m1, j2, m2, j, m):
    """Squared modulus of CG coefficient"""
    return abs(cg(j1, m1, j2, m2, j, m)) ** 2.


def Xi(Jg, Je, Epsilon):
    """The Xi class 

    Parameters
    ----------
    Jg: the angular momentum of the ground state 
    Je: the angular momentum of the excited state 
    Epsilon: the polarization of light field, in sperical coordinate (-,0,+) 

    Returns
    -------
    The scaled matrix Xi, and the scaling factor
    """
    # although convention has it that CG coefficients are real
    # but the light field may be complex
    ret = np.zeros((2*Je+1, 2*Jg+1), dtype=np.complex128) 

    denom = max(abs(cg(Jg,Jg,1,q,Je,Je)) for q in [-1, 0, 1])

    # we don't expect Je/Jg to be on the order 100
    for i, me in enumerate(range(-Je, Je+1)):
        for j, mg in enumerate(range(-Jg, Jg+1)): 
            for q in [-1, 0, 1]:
                ret[i, j] += cg(Jg,mg,1,q,Je,me) * Epsilon[q + 1]
                
                
    return ret / denom, denom


if __name__ == '__main__':
    
    print(CG(1,1,1,-1,0,0).doit())
    print(CG(1,0,1,0,0,0).doit())
    print(CG(1,-1,1,1,0,0).doit())
    print()
    print(CG(1,1,0,0,1,1).doit())
    print(CG(1,0,0,0,1,0).doit())
    print(CG(1,-1,0,0,1,-1).doit())
    
    print(Xi(0, 1, (0,1,0)))