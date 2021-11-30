from quantum_mcwf import Simulator
from multiprocessing import Pool, freeze_support
from tqdm import tqdm
import numpy as np


def worker(i, j):
    return simulators[j][i % pool_size].simulate(400000, 5e-4, [Simulator.momentum, Simulator.momentum2])


pool_size = 4
tot_sim = 10
ss = [20, 25, 16, 8, 40, 20]
deltas = [-10, -10, -8, -4, -20, -20]
simulators = [[Simulator(s, delta, 0, nmax=6)
               for _ in range(pool_size)] for s, delta in zip(ss, deltas)]

if __name__ == '__main__':
    freeze_support()

    for j, s, delta in zip(range(len(ss)), ss, deltas):
        with Pool(pool_size) as p, tqdm(total=tot_sim) as pbar:
            res = [p.apply_async(
                worker, args=(i, j), callback=lambda _: pbar.update(1)) for i in range(tot_sim)]
            results = np.array([r.get() for r in res])

        np.savetxt('s%sdelta%s_momentum.csv' %
                   (s, delta), results[..., 1].real, delimiter=',')
        np.savetxt('s%sdelta%s_momentum2.csv' %
                   (s, delta), results[..., 2].real, delimiter=',')
