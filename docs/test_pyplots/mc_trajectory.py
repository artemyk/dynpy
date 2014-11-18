import matplotlib.pyplot as plt
import dynpy

G = dynpy.sample_nets.karateclub_net
N = G.shape[0]
rw = dynpy.graphdynamics.RandomWalkerEnsemble(graph=G)

initState = np.zeros(N)
initState[ 5 ] = 1

trajectory = rw.get_trajectory(start_state=initState, max_time=30)
plt.imshow(trajectory, interpolation='none')
plt.xlabel('Node')
plt.ylabel('Time')
