import matplotlib.pyplot as plt
import numpy as np
import dynpy

G = dynpy.sample_nets.karateclub_net
N = G.shape[0]
rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=G, discrete_time=False)

initState = np.zeros(N)
initState[ 5 ] = 1

trajectory = rw.get_trajectory(start_state=initState, max_time=30)
plt.imshow(trajectory, interpolation='none')
plt.xlabel('Node')
plt.ylabel('Time')
