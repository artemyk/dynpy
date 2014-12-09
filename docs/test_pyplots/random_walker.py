import matplotlib.pyplot as plt
import numpy as np
import dynpy

num_steps = 30
G = dynpy.sample_nets.karateclub_net
N = G.shape[0]
rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=G)
sampler = dynpy.markov.MarkovChainSampler(rw)

# Initialize with a single random walker on node id=5
cState = np.zeros(N)
cState[ 5 ] = 1

spacetime = sampler.get_trajectory(start_state=cState, max_time=num_steps)

plt.spy(spacetime)
plt.xlabel('Node')
plt.ylabel('Time')
