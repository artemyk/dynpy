import matplotlib ; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import dynpy

kc = dynpy.sample_nets.karateclub_net
rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=kc, discrete_time=False)

eq_state = rw.get_equilibrium_distribution()
plt.imshow(np.atleast_2d(eq_state), interpolation='none')
