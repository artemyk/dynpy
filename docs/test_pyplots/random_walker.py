import matplotlib.pyplot as plt
import numpy as np
import dynpy

num_steps = 30
G = dynpy.sample_nets.karateclub_net
N = G.shape[0]
rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=G)
sampler = dynpy.markov.MarkovChainSampler(rw)

# Initialize with a single random walker on node id=5
trajectory = sampler.get_trajectory(start_state=5, max_time=num_steps)

plt.plot(np.arange(num_steps), trajectory, 'o')
plt.ylim([0, rw.num_states])
plt.xlabel('Time')
plt.ylabel('State')
