import matplotlib.pyplot as plt
import dynpy

bn = dynpy.bn.BooleanNetwork(rules=dynpy.sample_nets.budding_yeast_bn)
bnMC = dynpy.markov.MarkovChain.from_deterministic_system(bn)

# get distribution over states at various timepoints
t = bnMC.get_trajectory(start_state=bnMC.get_uniform_distribution(), max_time=20)

# project back from states onto activations of original nodes
bnProbs = t.dot(bn.get_ndx2state_mx())

# plot
plt.imshow(bnProbs, interpolation='none')
plt.xlabel('Node')
plt.ylabel('Time')